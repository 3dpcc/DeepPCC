import torch
import MinkowskiEngine as ME

from autoencoder import Encoder, Decoder
from models.entropy_model import EntropyBottleneck
from models.basic_module import HyperEncoder, HyperDecoder, ContextModelBase, ContextModelHyper
from models.entropy_model import EntropyBottleneck, SymmetricConditional
class PCCModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(channels=[1,16,32,64,32,8])
        self.decoder = Decoder(channels=[8,64,32,16])
        self.entropy_bottleneck = EntropyBottleneck(8)

    def get_likelihood(self, data, quantize_mode):
        data_F, likelihood = self.entropy_bottleneck(data.F,
            quantize_mode=quantize_mode)
        data_Q = ME.SparseTensor(
            features=data_F, 
            coordinate_map_key=data.coordinate_map_key, 
            coordinate_manager=data.coordinate_manager, 
            device=data.device)

        return data_Q, likelihood

    def forward(self, x, training=True):
        # Encoder
        # print(x.shape)
        # with torch.no_grad():
        y_list = self.encoder(x)
        y = y_list[0]
        # print('8888888888888888888888888888888')
        # print(y)
        # print(f'-----------------------y_lisy[0].shape{y_list[0].shape}')
        #print(y)
        #print('----------------------')
        ground_truth_list = y_list[1:] + [x]
        #print('yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
        #print(y_list[1].shape)

        #print(ground_truth_list)

        # nums (the number of ground truth)
        nums_list = [[len(C) for C in ground_truth.decomposed_coordinates] \
            for ground_truth in ground_truth_list]



        # Quantizer & Entropy Model
        # print(y)
        # with torch.no_grad():
        y_q, likelihood = self.get_likelihood(y,
                quantize_mode="noise" if training else "symbols")

        # Decoder
        out_cls_list, out = self.decoder(y_q, nums_list, ground_truth_list, training)

        # print(f'----------------likelihood.shape:{likelihood.shape}')

        return {'out':out,
                'out_cls_list':out_cls_list,
                'prior':y_q, 
                'likelihood':likelihood, 
                'ground_truth_list':ground_truth_list}

# class PCCModel_HyperContext(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#
#         self.encoder = Encoder(channels=[1,16,32,64,32,8])
#         self.decoder = Decoder(channels=[8,64,32,16])
#         # self.entropy_bottleneck = EntropyBottleneck(8)
#         self.hyper_encoder = HyperEncoder(32)
#         self.hyper_decoder = HyperDecoder(32)
#         self.new_entropy_bottleneck = EntropyBottleneck(8)
#         self.context_model = ContextModelHyper(8)
#         self.conditional_entropy_model = SymmetricConditional()
#
#
#     def get_likelihood(self, data, quantize_mode):
#         data_F, likelihood = self.entropy_bottleneck(data.F,
#             quantize_mode=quantize_mode)
#         data_Q = ME.SparseTensor(
#             features=data_F,
#             coordinate_map_key=data.coordinate_map_key,
#             coordinate_manager=data.coordinate_manager,
#             device=data.device)
#
#         return data_Q, likelihood
#
#     def forward(self, x, training=True):
#         # Encoder
#         # print(x.shape)
#         with torch.no_grad():
#             y_list = self.encoder(x)
#         y = y_list[0]
#         # print('8888888888888888888888888888888')
#         # print(y)
#         # print(f'-----------------------y_lisy[0].shape{y_list[0].shape}')
#         #print(y)
#         #print('----------------------')
#         ground_truth_list = y_list[1:] + [x]
#
#         z = self.hyper_encoder(y)
#         z_F, z_likelihood = self.new_entropy_bottleneck(z.F, quantize_mode="noise" if training else "symbols")
#         z_tilde = ME.SparseTensor(features=z_F, coordinate_map_key=z.coordinate_map_key,
#                                   coordinate_manager=z.coordinate_manager, device=z.device)
#
#         prior = self.hyper_decoder(z_tilde)
#         y_F = self.conditional_entropy_model._quantize(y.F, mode="noise" if training else "symbols")
#         y_tilde = ME.SparseTensor(features=y_F, coordinate_map_key=y.coordinate_map_key,
#                                   coordinate_manager=y.coordinate_manager, device=y.device)
#
#         loc, scale = self.context_model(y_tilde, prior)
#         scale = torch.clamp(scale, min=1e-8)  # lower_bound
#         # conditional entropy model.
#         _, y_likelihood = self.conditional_entropy_model(y_F, loc, scale, quantize_mode=None)
#
#         # nums (the number of ground truth)
#         nums_list = [[len(C) for C in ground_truth.decomposed_coordinates] \
#             for ground_truth in ground_truth_list]
#
#
#         # Quantizer & Entropy Model
#         # print(y)
#
#         # Decoder
#         with torch.no_grad():
#             out_cls_list, out = self.decoder(y_tilde, nums_list, ground_truth_list, training)
#
#         # print(f'----------------likelihood.shape:{likelihood.shape}')
#
#         return {'out':out,
#                 'out_cls_list':out_cls_list,
#                 'prior':y_tilde,
#                 'z_likelihood':z_likelihood,
#                 'y_likelihood':y_likelihood,
#                 'ground_truth_list':ground_truth_list}
#     @torch.no_grad
#     def encode(self,y):
#         z = self.hyper_encoder(y)
#
#         z_F = self.entropy_bottleneck._quantize(z.F, mode="symbols")
#         z_tilde = ME.SparseTensor(features=z_F, coordinate_map_key=z.coordinate_map_key,
#                                   coordinate_manager=z.coordinate_manager, device=z.device)
#         prior = self.hyper_decoder(z_tilde)  # extimate prior from z
#         # encode z
#         z = sort_sparse_tensor(z)  # resort
#         z_strings, z_min_v, z_max_v = self.entropy_bottleneck.compress(z.F)
#         z_shape = z.F.shape
#         with open(filename + '_z_F.bin', 'wb') as fout:
#             fout.write(z_strings)
#         with open(filename + '_z_H.bin', 'wb') as fout:
#             fout.write(np.array(z_shape, dtype=np.int32).tobytes())
#             fout.write(np.array(len(z_min_v), dtype=np.int8).tobytes())
#             fout.write(np.array(z_min_v, dtype=np.float32).tobytes())
#             fout.write(np.array(z_max_v, dtype=np.float32).tobytes())
#         z_Bytes = os.path.getsize(filename + '_z_F.bin') + os.path.getsize(filename + '_z_H.bin')
#
#         # encode y
#         y = sort_sparse_tensor(y)  # resort
#         prior = sort_sparse_tensor(prior)  # resort
#         loc = prior.F[:, :prior.F.shape[-1] // 2]
#         scale = prior.F[:, prior.F.shape[-1] // 2:].abs()
#         scale = torch.clamp(scale, min=1e-8)  # lower_bound
#         y_strings, y_min_v, y_max_v = self.conditional_entropy_model.compress(y.F, loc, scale)
#         y_shape = y.F.shape
#         with open(filename + '_y_F.bin', 'wb') as fout:
#             fout.write(y_strings)
#         with open(filename + '_y_H.bin', 'wb') as fout:
#             fout.write(np.array(y_shape, dtype=np.int32).tobytes())
#             # fout.write(np.array(len(y_min_v), dtype=np.int8).tobytes())
#             fout.write(np.array(y_min_v, dtype=np.float32).tobytes())
#             fout.write(np.array(y_max_v, dtype=np.float32).tobytes())
#         y_Bytes = os.path.getsize(filename + '_y_F.bin') + os.path.getsize(filename + '_y_H.bin')
#
#     @torch.no_grad
#     def decode(self, y):
#         downsamplerB = torch.nn.Sequential(*[ME.MinkowskiMaxPooling(
#             kernel_size=2, stride=2, dimension=3)] * 2).to(device)
#         z = downsamplerB(y)
#
#         # decode z
#         with open(filename + '_z_F.bin', 'rb') as fin:
#             z_strings = fin.read()
#         with open(filename + '_z_H.bin', 'rb') as fin:
#             z_shape = np.frombuffer(fin.read(4 * 2), dtype=np.int32)
#             len_min_v = np.frombuffer(fin.read(1), dtype=np.int8)[0]
#             z_min_v = np.frombuffer(fin.read(4 * len_min_v), dtype=np.float32)
#             z_max_v = np.frombuffer(fin.read(4 * len_min_v), dtype=np.float32)
#         z_F = self.entropy_bottleneck.decompress(z_strings, z_min_v, z_max_v, z_shape, channels=z_shape[-1])
#
#         index_z = array2vector(z.C.cpu().numpy()).argsort().argsort()
#
#         z = ME.SparseTensor(features=z_F[index_z],
#                             coordinate_map_key=z.coordinate_map_key,
#                             coordinate_manager=z.coordinate_manager,
#                             device=z.device)
#         # decode y
#         prior = self.hyper_decoder(z)
#         prior = sort_sparse_tensor(prior)  # resort
#         loc = prior.F[:, :prior.F.shape[-1] // 2]
#         scale = prior.F[:, prior.F.shape[-1] // 2:].abs()
#         scale = torch.clamp(scale, min=1e-8)  # lower_bound
#         with open(filename + '_y_F.bin', 'rb') as fin:
#             y_strings = fin.read()
#         with open(filename + '_y_H.bin', 'rb') as fin:
#             y_shape = np.frombuffer(fin.read(4 * 2), dtype=np.int32)
#             y_min_v = np.frombuffer(fin.read(4), dtype=np.float32)[0]
#             y_max_v = np.frombuffer(fin.read(4), dtype=np.float32)[0]
#         y_F = self.conditional_entropy_model.decompress(y_strings, loc, scale, y_min_v, y_max_v, y_shape,
#                                                         channels=y_shape[-1])
#         index_y = array2vector(y.C.cpu().numpy()).argsort().argsort()
#         y = ME.SparseTensor(features=y_F[index_y],
#                             coordinate_map_key=y.coordinate_map_key,
#                             coordinate_manager=y.coordinate_manager,
#                             device=y.device)
#         return y

if __name__ == '__main__':
    model = PCCModel()
    print(model)

