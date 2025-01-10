import os
import numpy as np
import torch
import MinkowskiEngine as ME
from models.basic_module import Encoder, Decoder, HyperEncoder, HyperDecoder, ContextModelHyper, \
    Enhancer
from models.entropy_model import EntropyBottleneck, SymmetricConditional
from data_processing.data_utils import sort_sparse_tensor, array2vector, rgb2yuv, yuv2rgb
from data_processing.data_utils import write_ply_o3d_geo, read_ply_o3d_geo, kdtree_partition
from extension.gpcc import gpcc_encode, gpcc_decode
from models.y_module import Enhancer
from models.u_module import Enhancer_uv



class PCACModelContextHyperGuided(torch.nn.Module):
    def __init__(self, channels=128, A=16):
        super().__init__()
        self.A = A
        self.encoder = Encoder(channels)
        self.decoder = Decoder(channels)
        self.hyper_encoder = HyperEncoder(channels)
        self.hyper_decoder = HyperDecoder(channels)
        self.entropy_bottleneck = EntropyBottleneck(channels)
        self.context_model = ContextModelHyper(channels)
        self.conditional_entropy_model = SymmetricConditional()
        self.module_y = Enhancer(channels=channels, A=A)
        self.module_uv = Enhancer_uv(channels, A=A)


    def forward(self, x, training, partition=False, num_points=0):
        y = self.encoder(x)
        z = self.hyper_encoder(y)
        z_F, z_likelihood = self.entropy_bottleneck(z.F, quantize_mode="noise" if training else "symbols")
        z_tilde = ME.SparseTensor(features=z_F, coordinate_map_key=z.coordinate_map_key,
                                  coordinate_manager=z.coordinate_manager, device=z.device)
        prior = self.hyper_decoder(z_tilde)
        y_F = self.conditional_entropy_model._quantize(y.F, mode="noise" if training else "symbols")
        y_tilde = ME.SparseTensor(features=y_F, coordinate_map_key=y.coordinate_map_key,
                                  coordinate_manager=y.coordinate_manager, device=y.device)
        loc, scale = self.context_model(y_tilde, prior)
        scale = torch.clamp(scale, min=1e-8)  # lower_bound
        # conditional entropy model.
        _, y_likelihood = self.conditional_entropy_model(y_F, loc, scale, quantize_mode=None)
        out = self.decoder(y_tilde)
        max = ME.MinkowskiMaxPooling(kernel_size=1,stride=1,dilation=1,dimension=3)
        if partition:
            gt_coords = max(x, out.C).C[:,1:].cpu().numpy()
            out_coords = out.C[:, 1:].cpu().numpy()
            gt_feats = max(x, out.C).F.cpu().numpy()
            out_feats = out.F.cpu().numpy()
            gt_pcs = np.concatenate([gt_coords, gt_feats], axis=1)
            out_pcs = np.concatenate([out_coords, out_feats], axis=1)
            kdtree_gt_parts = kdtree_partition(gt_pcs, max_num=num_points)
            kdtree_out_parts = kdtree_partition(out_pcs, max_num=num_points)
            for idx, out_pc in enumerate(kdtree_out_parts):
                gt_pc = kdtree_gt_parts[idx]
                gt_coord, gt_feat = ME.utils.sparse_collate([torch.tensor(gt_pc[:,:3]).int()], [torch.tensor(gt_pc[:,3:]).float()])
                out_coord, out_feat = ME.utils.sparse_collate([torch.tensor(out_pc[:, :3]).int()], [torch.tensor(out_pc[:, 3:]).float()])
                gt = ME.SparseTensor(features=gt_feat, coordinates=gt_coord, device=x.device)
                yuv = rgb2yuv(out_feat)
                y = yuv[:, :1].clone().cuda()
                true_y = rgb2yuv(gt.F)[:, :1]
                guided_input = ME.SparseTensor(features=y, coordinates=out_coord, device=x.device)
                out_features = self.module_y(guided_input)
                r_out = out_features.F
                true_y = true_y.cuda()
                b_feature = torch.subtract(true_y, guided_input.F)
                A = torch.matmul(
                    torch.matmul(torch.inverse(torch.matmul(r_out.permute(1, 0), r_out)), r_out.permute(1, 0)),
                    b_feature)
                print(A)
                b_feature = b_feature.permute(1, 0)
                loss = torch.sum(-(torch.matmul(torch.matmul(b_feature, r_out), A)))
                print(loss)
                with torch.no_grad():
                    residual = torch.matmul(r_out, A)
                    if idx == 0:
                        new_y = residual + y
                        out_coords = out_coord
                    else:
                        out_coords = torch.cat((out_coords, out_coord), dim=0)
                        new_y = torch.cat((new_y, residual + y), dim=0)
            endtoend_out_yuv = rgb2yuv(out.F)
            endtoend_out_yuv[:,:1] = new_y
            enhancer_out = ME.SparseTensor(features=yuv2rgb(endtoend_out_yuv), coordinates=out_coords, device=x.device)
        else:
            yuv = rgb2yuv(out.F)
            y = yuv[:, :1]
            true_y = rgb2yuv(x.F)[:, :1]
            guided_input = ME.SparseTensor(features=y, coordinates=x.C, device=x.device)
            out_features = self.module_y(guided_input)
            r_out = out_features.F
            true_y = true_y.cuda()
            b_feature = torch.subtract(true_y, guided_input.F)
            A = torch.matmul(
                torch.matmul(torch.inverse(torch.matmul(r_out.permute(1, 0), r_out)), r_out.permute(1, 0)),
                b_feature)
            print(A)
            b_feature = b_feature.permute(1, 0)
            loss = torch.sum(-(torch.matmul(torch.matmul(b_feature, r_out), A)))

            with torch.no_grad():
                residual = torch.matmul(r_out, A)
                yuv[:, :1] = residual + y

            uv = yuv[:, 1:]
            true_uv = rgb2yuv(x.F)[:, 1:]

            guided_y = ME.SparseTensor(features=y, coordinates=x.C, device=x.device)
            guided_input = ME.SparseTensor(features=uv, coordinates=x.C, device=x.device)
            out_features = self.module_uv(guided_y, guided_input)
            r_out = out_features.F.reshape(-1, self.A)
            true_uv = true_uv.cuda()
            b_feature = torch.subtract(true_uv, guided_input.F).reshape(-1, 1)
            A = torch.matmul(
                torch.matmul(torch.inverse(torch.matmul(r_out.permute(1, 0), r_out)), r_out.permute(1, 0)),
                b_feature)
            print(A)
            b_feature = b_feature.permute(1, 0)

            loss = torch.sum(-(torch.matmul(torch.matmul(b_feature, r_out), A)))
            with torch.no_grad():
                residual = torch.matmul(r_out, A).reshape(-1, 2)
                yuv[:, 1:] = residual + guided_input.F

            enhancer_out = ME.SparseTensor(features=yuv2rgb(yuv), coordinates=x.C, device=x.device)

        return {'likelihood': y_likelihood,
                'y_tilde': y_tilde,
                'z_likelihood': z_likelihood,
                'z_tilde': z_tilde,
                'out': out,
                'x': x,
                'guided_loss': loss,
                'enhancer_sparsetensor': enhancer_out}

    @torch.no_grad()
    def encode(self, x, filename):
        x = sort_sparse_tensor(x)
        y = self.encoder(x)
        z = self.hyper_encoder(y)
        z_F = self.entropy_bottleneck._quantize(z.F, mode="symbols")
        z_tilde = ME.SparseTensor(features=z_F, coordinate_map_key=z.coordinate_map_key,
                                  coordinate_manager=z.coordinate_manager, device=z.device)
        prior = self.hyper_decoder(z_tilde)  # estimate prior from z
        # encode z
        z = sort_sparse_tensor(z)  # resort
        z_strings, z_min_v, z_max_v = self.entropy_bottleneck.compress(z.F)
        z_shape = z.F.shape
        with open(filename + '_z_F.bin', 'wb') as fout:
            fout.write(z_strings)
        with open(filename + '_z_H.bin', 'wb') as fout:
            fout.write(np.array(z_shape, dtype=np.int32).tobytes())
            fout.write(np.array(len(z_min_v), dtype=np.int8).tobytes())
            fout.write(np.array(z_min_v, dtype=np.float32).tobytes())
            fout.write(np.array(z_max_v, dtype=np.float32).tobytes())
        z_Bytes = os.path.getsize(filename + '_z_F.bin') + os.path.getsize(filename + '_z_H.bin')
        # encode y
        y_F = self.conditional_entropy_model._quantize(y.F, mode="symbols")
        y_tilde = ME.SparseTensor(features=y_F, coordinate_map_key=y.coordinate_map_key,
                                  coordinate_manager=y.coordinate_manager, device=y.device)
        y_tilde = sort_sparse_tensor(y_tilde)  # resort
        prior = sort_sparse_tensor(prior)  # resort

        # conditional entropy model.
        loc, scale = self.context_model(y_tilde, prior)
        scale = torch.clamp(scale, min=1e-8)  # lower_bound
        # estimate bitrate
        if False:
            _, likelihood = self.conditional_entropy_model(y_tilde.F, loc, scale, quantize_mode=None)
            bpp = -torch.sum(torch.log2(likelihood)).cpu().numpy() / float(x.__len__())
            print('estimated bpp:', round(bpp, 4))

        y_strings, y_min_v, y_max_v = self.conditional_entropy_model.compress_context(y_tilde.F, loc, scale)
        y_shape = y.F.shape
        with open(filename + '_y_F.bin', 'wb') as fout:
            fout.write(y_strings)
        with open(filename + '_y_H.bin', 'wb') as fout:
            fout.write(np.array(y_shape, dtype=np.int32).tobytes())
            # fout.write(np.array(len(y_min_v), dtype=np.int8).tobytes())
            fout.write(np.array(y_min_v, dtype=np.float32).tobytes())
            fout.write(np.array(y_max_v, dtype=np.float32).tobytes())
        y_Bytes = os.path.getsize(filename + '_y_F.bin') + os.path.getsize(filename + '_y_H.bin')
        # encode coords
        x_C = x.C.cpu().numpy()[:, 1:].astype('int')
        write_ply_o3d_geo(filedir=filename + '_C.ply', coords=x_C)
        _ = gpcc_encode(filename + '_C.ply', filename + '_C.bin',
                        posQuantscale=1, qp=None)
        y_tilde = ME.SparseTensor(features=y_F, coordinate_map_key=y.coordinate_map_key,
                                  coordinate_manager=y.coordinate_manager, device=y.device)
        out = self.decoder(y_tilde)
        yuv = rgb2yuv(out.F)
        y = yuv[:, :1]
        true_y = rgb2yuv(x.F)[:, :1]
        guided_input = ME.SparseTensor(features=y, coordinates=out.C, device=x.device)
        out_features = self.module_y(guided_input)
        r_out = out_features.F
        true_y = true_y.cuda()
        b_feature = torch.subtract(true_y, guided_input.F)
        A_Y = torch.matmul(
            torch.matmul(torch.inverse(torch.matmul(r_out.permute(1, 0), r_out)), r_out.permute(1, 0)),
            b_feature)

        uv = yuv[:, 1:]
        true_uv = rgb2yuv(x.F)[:, 1:]

        guided_y = ME.SparseTensor(features=y, coordinates=x.C, device=x.device)
        guided_input = ME.SparseTensor(features=uv, coordinates=x.C, device=x.device)
        out_features = self.module_uv(guided_y, guided_input)
        r_out = out_features.F.reshape(-1, self.A)
        true_uv = true_uv.cuda()
        b_feature = torch.subtract(true_uv, guided_input.F).reshape(-1, 1)
        A_UV = torch.matmul(
            torch.matmul(torch.inverse(torch.matmul(r_out.permute(1, 0), r_out)), r_out.permute(1, 0)),
            b_feature)
        A_Y_Q = torch.round(A_Y * 1000)
        A_UV_Q = torch.round(A_UV * 1000)

        return z_Bytes + y_Bytes, A_Y_Q, A_UV_Q

    @torch.no_grad()
    def decode(self, filename, A_Y_Q, A_UV_Q, device):
        # decode coords
        _ = gpcc_decode(filename + '_C.bin', filename + '_C_dec.ply', attr=False)
        x_C = read_ply_o3d_geo(filename + '_C_dec.ply')
        x_C, x_F = ME.utils.sparse_collate([x_C], [torch.zeros([len(x_C), 1])])
        x = ME.SparseTensor(features=x_F.float(), coordinates=x_C.int(), device=device)
        x = sort_sparse_tensor(x)
        # y & z coords
        downsamplerA = torch.nn.Sequential(*[ME.MinkowskiMaxPooling(
            kernel_size=2, stride=2, dimension=3)] * 3).to(device)
        downsamplerB = torch.nn.Sequential(*[ME.MinkowskiMaxPooling(
            kernel_size=2, stride=2, dimension=3)] * 2).to(device)
        y = downsamplerA(x)
        z = downsamplerB(y)
        # decode z
        with open(filename + '_z_F.bin', 'rb') as fin:
            z_strings = fin.read()
        with open(filename + '_z_H.bin', 'rb') as fin:
            z_shape = np.frombuffer(fin.read(4 * 2), dtype=np.int32)
            len_min_v = np.frombuffer(fin.read(1), dtype=np.int8)[0]
            z_min_v = np.frombuffer(fin.read(4 * len_min_v), dtype=np.float32)
            z_max_v = np.frombuffer(fin.read(4 * len_min_v), dtype=np.float32)
        z_F = self.entropy_bottleneck.decompress(z_strings, z_min_v, z_max_v, z_shape, channels=z_shape[-1])
        index_z = array2vector(z.C.cpu().numpy()).argsort().argsort()
        z = ME.SparseTensor(features=z_F[index_z],
                            coordinate_map_key=z.coordinate_map_key,
                            coordinate_manager=z.coordinate_manager,
                            device=z.device)
        # get prior
        prior = self.hyper_decoder(z)
        prior = sort_sparse_tensor(prior)  # resort
        # decode y
        with open(filename + '_y_H.bin', 'rb') as fin:
            y_shape = np.frombuffer(fin.read(4 * 2), dtype=np.int32)
            y_min_v = np.frombuffer(fin.read(4), dtype=np.float32)[0]
            y_max_v = np.frombuffer(fin.read(4), dtype=np.float32)[0]
        with open(filename + '_y_F.bin', 'rb') as fin:
            y_strings = fin.read()
        # palceholder
        y_dec = sort_sparse_tensor(y)
        y_dec = ME.SparseTensor(features=torch.zeros([y_shape[0], y_shape[1]]).float(),
                                coordinate_map_key=y_dec.coordinate_map_key,
                                coordinate_manager=y_dec.coordinate_manager,
                                device=y_dec.device)
        y_dec = self.conditional_entropy_model.decompress_context(self.context_model, y_dec, y_strings, y_min_v,
                                                                  y_max_v,
                                                                  prior=prior)
        index_y = array2vector(y.C.cpu().numpy()).argsort().argsort()
        y = ME.SparseTensor(features=y_dec.F[index_y],
                            coordinate_map_key=y.coordinate_map_key,
                            coordinate_manager=y.coordinate_manager,
                            device=y.device)
        x = self.decoder(y)
        yuv = rgb2yuv(x.F)
        y = yuv[:, :1]
        guided_input = ME.SparseTensor(features=y, coordinates=x.C, device=x.device)
        out_features = self.module_y(guided_input)
        r_out = out_features.F
        residual = torch.matmul(r_out, A_Y_Q / 1000)
        yuv[:, :1] = residual + y
        uv = yuv[:, 1:]
        guided_y = ME.SparseTensor(features=y, coordinates=x.C, device=x.device)
        guided_input = ME.SparseTensor(features=uv, coordinates=x.C, device=x.device)
        out_features = self.module_uv(guided_y, guided_input)
        r_out = out_features.F.reshape(-1, self.A)
        residual = torch.matmul(r_out, A_UV_Q / 1000).reshape(-1, 2)
        yuv[:, 1:] = residual + guided_input.F
        enhancer_out = ME.SparseTensor(features=yuv2rgb(yuv), coordinates=x.C, device=x.device)

        return enhancer_out


class PCACModelContextHyperGuided_UV(torch.nn.Module):
    def __init__(self, channels=128, A=16):
        super().__init__()
        self.encoder = Encoder(channels)
        self.decoder = Decoder(channels)
        self.hyper_encoder = HyperEncoder(channels)
        self.hyper_decoder = HyperDecoder(channels)
        self.entropy_bottleneck = EntropyBottleneck(channels)
        self.context_model = ContextModelHyper(channels)
        self.conditional_entropy_model = SymmetricConditional()
        self.module_uv = Enhancer_uv(channels, A)
        self.A = A

    def forward(self, x, training):
        y = self.encoder(x)
        z = self.hyper_encoder(y)
        z_F, z_likelihood = self.entropy_bottleneck(z.F, quantize_mode="noise" if training else "symbols")
        z_tilde = ME.SparseTensor(features=z_F, coordinate_map_key=z.coordinate_map_key,
                                  coordinate_manager=z.coordinate_manager, device=z.device)
        prior = self.hyper_decoder(z_tilde)
        y_F = self.conditional_entropy_model._quantize(y.F, mode="noise" if training else "symbols")
        y_tilde = ME.SparseTensor(features=y_F, coordinate_map_key=y.coordinate_map_key,
                                  coordinate_manager=y.coordinate_manager, device=y.device)
        loc, scale = self.context_model(y_tilde, prior)
        scale = torch.clamp(scale, min=1e-8)  # lower_bound
        # conditional entropy model.
        _, y_likelihood = self.conditional_entropy_model(y_F, loc, scale, quantize_mode=None)
        out = self.decoder(y_tilde)

        yuv = rgb2yuv(out.F)
        y = yuv[:, :1]
        uv = yuv[:, 1:]
        true_uv = rgb2yuv(x.F)[:, 1:]

        guided_y = ME.SparseTensor(features=y, coordinates=x.C, device=x.device)
        guided_input = ME.SparseTensor(features=uv, coordinates=x.C, device=x.device)
        out_features = self.module_uv(guided_y, guided_input)
        r_out = out_features.F.reshape(-1, self.A)
        true_uv = true_uv.cuda()
        b_feature = torch.subtract(true_uv, guided_input.F).reshape(-1, 1)
        A = torch.matmul(
            torch.matmul(torch.inverse(torch.matmul(r_out.permute(1, 0), r_out)), r_out.permute(1, 0)),
            b_feature)
        print(A)
        b_feature = b_feature.permute(1, 0)

        loss = torch.sum(-(torch.matmul(torch.matmul(b_feature, r_out), A)))
        with torch.no_grad():
            residual = torch.matmul(r_out, A).reshape(-1, 2)
            yuv[:, 1:] = residual + guided_input.F
            enhancer_out = ME.SparseTensor(features=yuv2rgb(yuv), coordinates=x.C, device=x.device)

        return {'likelihood': y_likelihood,
                'y_tilde': y_tilde,
                'z_likelihood': z_likelihood,
                'z_tilde': z_tilde,
                'out': out,
                'x': x,
                'guided_loss': loss,
                'enhancer_sparsetensor': enhancer_out}

    @torch.no_grad()
    def encode(self, x, filename):
        x = sort_sparse_tensor(x)
        y = self.encoder(x)
        z = self.hyper_encoder(y)
        z_F = self.entropy_bottleneck._quantize(z.F, mode="symbols")
        z_tilde = ME.SparseTensor(features=z_F, coordinate_map_key=z.coordinate_map_key,
                                  coordinate_manager=z.coordinate_manager, device=z.device)
        prior = self.hyper_decoder(z_tilde)  # estimate prior from z
        # encode z
        z = sort_sparse_tensor(z)  # resort
        z_strings, z_min_v, z_max_v = self.entropy_bottleneck.compress(z.F)
        z_shape = z.F.shape
        with open(filename + '_z_F.bin', 'wb') as fout:
            fout.write(z_strings)
        with open(filename + '_z_H.bin', 'wb') as fout:
            fout.write(np.array(z_shape, dtype=np.int32).tobytes())
            fout.write(np.array(len(z_min_v), dtype=np.int8).tobytes())
            fout.write(np.array(z_min_v, dtype=np.float32).tobytes())
            fout.write(np.array(z_max_v, dtype=np.float32).tobytes())
        z_Bytes = os.path.getsize(filename + '_z_F.bin') + os.path.getsize(filename + '_z_H.bin')
        # encode y
        y_F = self.conditional_entropy_model._quantize(y.F, mode="symbols")
        y_tilde = ME.SparseTensor(features=y_F, coordinate_map_key=y.coordinate_map_key,
                                  coordinate_manager=y.coordinate_manager, device=y.device)
        y_tilde = sort_sparse_tensor(y_tilde)  # resort
        prior = sort_sparse_tensor(prior)  # resort

        # conditional entropy model.
        loc, scale = self.context_model(y_tilde, prior)
        scale = torch.clamp(scale, min=1e-8)  # lower_bound
        # estimate bitrate
        if False:
            _, likelihood = self.conditional_entropy_model(y_tilde.F, loc, scale, quantize_mode=None)
            bpp = -torch.sum(torch.log2(likelihood)).cpu().numpy() / float(x.__len__())
            print('estimated bpp:', round(bpp, 4))

        y_strings, y_min_v, y_max_v = self.conditional_entropy_model.compress_context(y_tilde.F, loc, scale)
        y_shape = y.F.shape
        with open(filename + '_y_F.bin', 'wb') as fout:
            fout.write(y_strings)
        with open(filename + '_y_H.bin', 'wb') as fout:
            fout.write(np.array(y_shape, dtype=np.int32).tobytes())
            # fout.write(np.array(len(y_min_v), dtype=np.int8).tobytes())
            fout.write(np.array(y_min_v, dtype=np.float32).tobytes())
            fout.write(np.array(y_max_v, dtype=np.float32).tobytes())
        y_Bytes = os.path.getsize(filename + '_y_F.bin') + os.path.getsize(filename + '_y_H.bin')
        # encode coords
        x_C = x.C.cpu().numpy()[:, 1:].astype('int')
        write_ply_o3d_geo(filedir=filename + '_C.ply', coords=x_C)
        _ = gpcc_encode(filename + '_C.ply', filename + '_C.bin',
                        posQuantscale=1, qp=None)

        return z_Bytes + y_Bytes

    @torch.no_grad()
    def decode(self, filename, device):
        # decode coords
        _ = gpcc_decode(filename + '_C.bin', filename + '_C_dec.ply', attr=False)
        x_C = read_ply_o3d_geo(filename + '_C_dec.ply')
        x_C, x_F = ME.utils.sparse_collate([x_C], [torch.zeros([len(x_C), 1])])
        x = ME.SparseTensor(features=x_F.float(), coordinates=x_C.int(), device=device)
        x = sort_sparse_tensor(x)
        # y & z coords
        downsamplerA = torch.nn.Sequential(*[ME.MinkowskiMaxPooling(
            kernel_size=2, stride=2, dimension=3)] * 3).to(device)
        downsamplerB = torch.nn.Sequential(*[ME.MinkowskiMaxPooling(
            kernel_size=2, stride=2, dimension=3)] * 2).to(device)
        y = downsamplerA(x)
        z = downsamplerB(y)
        # decode z
        with open(filename + '_z_F.bin', 'rb') as fin:
            z_strings = fin.read()
        with open(filename + '_z_H.bin', 'rb') as fin:
            z_shape = np.frombuffer(fin.read(4 * 2), dtype=np.int32)
            len_min_v = np.frombuffer(fin.read(1), dtype=np.int8)[0]
            z_min_v = np.frombuffer(fin.read(4 * len_min_v), dtype=np.float32)
            z_max_v = np.frombuffer(fin.read(4 * len_min_v), dtype=np.float32)
        z_F = self.entropy_bottleneck.decompress(z_strings, z_min_v, z_max_v, z_shape, channels=z_shape[-1])
        index_z = array2vector(z.C.cpu().numpy()).argsort().argsort()
        z = ME.SparseTensor(features=z_F[index_z],
                            coordinate_map_key=z.coordinate_map_key,
                            coordinate_manager=z.coordinate_manager,
                            device=z.device)
        # get prior
        prior = self.hyper_decoder(z)
        prior = sort_sparse_tensor(prior)  # resort
        # decode y
        with open(filename + '_y_H.bin', 'rb') as fin:
            y_shape = np.frombuffer(fin.read(4 * 2), dtype=np.int32)
            y_min_v = np.frombuffer(fin.read(4), dtype=np.float32)[0]
            y_max_v = np.frombuffer(fin.read(4), dtype=np.float32)[0]
        with open(filename + '_y_F.bin', 'rb') as fin:
            y_strings = fin.read()
        # palceholder
        y_dec = sort_sparse_tensor(y)
        y_dec = ME.SparseTensor(features=torch.zeros([y_shape[0], y_shape[1]]).float(),
                                coordinate_map_key=y_dec.coordinate_map_key,
                                coordinate_manager=y_dec.coordinate_manager,
                                device=y_dec.device)
        y_dec = self.conditional_entropy_model.decompress_context(self.context_model, y_dec, y_strings, y_min_v,
                                                                  y_max_v,
                                                                  prior=prior)
        index_y = array2vector(y.C.cpu().numpy()).argsort().argsort()
        y = ME.SparseTensor(features=y_dec.F[index_y],
                            coordinate_map_key=y.coordinate_map_key,
                            coordinate_manager=y.coordinate_manager,
                            device=y.device)
        x = self.decoder(y)

        return x


if __name__ == '__main__':
    model = PCACModelContextHyperGuided(channels=128, A=16)
    ckpt_y = './ckpts/2000_Y/epoch_last.pth'
    ckpt_uv = './ckpts/2000_UV/epoch_last.pth'
    ckpts_y = torch.load(ckpt_y)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in ckpts_y['model'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    ckpts_uv = torch.load(ckpt_uv)
    pretrained_dict = {k: v for k, v in ckpts_uv['model'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    torch.save({'model': model.state_dict()},
               'ckpts/r04.pth')
