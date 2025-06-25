import numpy as np
import torch
import os, glob, argparse, time
import pandas as pd
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from data_processing.data_utils import load_sparse_tensor, read_h5, write_ply_ascii, write_ply_ascii_all, read_ply_ascii_geo, write_ply_head, rgb2yuv,yuv2rgb, ycgcor2rgb, rgb2ycgcor, mse_y
from extension.pc_error import pc_error
import MinkowskiEngine as ME
from shutil import copyfile
from data_processing.data_utils import read_ply_ascii, kdtree_partition
import pytorch3d
import subprocess


def run_cmd(cmd):
    std_out_and_err = subprocess.Popen(cmd,
                     shell=True,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE).communicate()
    return std_out_and_err

def knn_interpolation(x, new_coords, k):
    new_coords_C = new_coords.unsqueeze(0).float()
    x_coords_C = x.C[:, 1:].unsqueeze(0).float()
    x_attr = x.F.unsqueeze(0).float()
    x_nn = pytorch3d.ops.knn_points(new_coords_C, x_coords_C, K=k)
    # knn_coords = pytorch3d.ops.knn_gather(x_coords_C, x_nn.idx).squeeze(0)
    knn_attribute = pytorch3d.ops.knn_gather(x_attr, x_nn.idx).squeeze(0)
    dists_sqrt = torch.sqrt(x_nn.dists)
    sigma = torch.mean(dists_sqrt, dim=2)
    # sigma = torch.std(dists_sqrt)
    sigma2 = sigma.unsqueeze(2) ** 2
    # sigma2 = sigma ** 2
    weight_ij = torch.exp(-(dists_sqrt / (2 * sigma2 + 1e-8))).squeeze(0)
    weight_ij_squeeze = weight_ij.unsqueeze(2)
    d_i = 1 / torch.sum(weight_ij, dim=1).unsqueeze(1)
    knn_attribute_weight = weight_ij_squeeze * knn_attribute

    interpolation_attribute = d_i * torch.sum(knn_attribute_weight, dim=1)
    return interpolation_attribute

def test_compression(model_list, input_rootdir, output_rootdir, input_format, quick, lossyGeo_rootdir):
    input_filedirs = sorted(glob.glob(os.path.join(input_rootdir, '**', f'*' + input_format), recursive=True))
    lossyGeo_filedirs = sorted(glob.glob(os.path.join(lossyGeo_rootdir, '**', f'*' + input_format), recursive=True))
    input_filedirs = input_filedirs[: args.frame]
    lossyGeo_filedirs = lossyGeo_filedirs[: args.frame]
    for idx_file, input_filedir in enumerate(tqdm(input_filedirs)):
        # load data
        filename = os.path.join(output_rootdir, input_filedir[len(input_rootdir):].split('.')[0].split('/')[-1])
        print(filename)
        out_folder, _ = os.path.split(filename);
        os.makedirs(out_folder, exist_ok=True)
        if input_filedir.endswith('ply'):
            filedir_ply = input_filedir
        elif input_filedir.endswith('h5'):
            coords, feats = read_h5(input_filedir)
            filedir_ply = os.path.join(output_rootdir, filename + '.ply')
            write_ply_ascii(filedir_ply, coords, feats)
        gt = load_sparse_tensor(filedir_ply, order='rgb')
        lossy_coords = read_ply_ascii_geo(lossyGeo_filedirs[idx_file])
        lossy_coords = torch.tensor(lossy_coords).cuda()
        knn_attribute = knn_interpolation(gt, lossy_coords, 1)
        lossy_coords, knn_attribute = ME.utils.sparse_collate([lossy_coords], [knn_attribute])
        x = ME.SparseTensor(features=knn_attribute, coordinates=lossy_coords, device=gt.device)

        for idx_rate, model in enumerate(model_list):
            if quick:
                # estimate bpp
                with torch.no_grad():
                    out_set = model(x, training=False)
                out = out_set['enhancer_sparsetensor']

                bpp = -torch.sum(torch.log2(out_set['likelihood'])).cpu().numpy() / float(x.__len__())
                results_one = {'bpp_att': bpp}
                if 'z_likelihood' in out_set:
                    z_bpp = - torch.sum(torch.log2(out_set['z_likelihood'])).cpu().numpy() / float(x.__len__())
                    results_one['bpp_att'] += z_bpp
                    results_one['bpp_hyper'] = z_bpp
                print('dbg: estimated bpp:\t', round(results_one['bpp_att'], 2))
            else:
                # real bpp
                start_time = time.time()
                Bytes, A_Y_Q, A_UV_Q = model.encode(x, filename + '_r' + str(idx_rate))
                enc_time = round(time.time() - start_time, 3)
                bpp = round(8 * Bytes / float(x.__len__()), 4)
                start_time = time.time()
                out = model.decode(filename + '_r' + str(idx_rate), A_Y_Q, A_UV_Q, device)
                dec_time = round(time.time() - start_time, 3)
                results_one = {'bpp_att': bpp, 'enc_time': enc_time, 'dec_time': dec_time}
                print('dbg: real bpp:\t', round(bpp, 2))

            # distortion
            coords = out.C.cpu().numpy()[:, 1:]
            feats = (out.F.detach().cpu().numpy() * 255).round()
            feats = np.clip(feats, 0, 255).astype('uint8')
            dec_dir = filename + '_' + 'r' + str(idx_rate) + '_dec.ply'
            from data_processing.data_utils import write_ply_ascii
            write_ply_ascii(dec_dir, coords, feats)
            pc_error_results = pc_error(filedir_ply, dec_dir,
                                        res=args.resolution, normal=True, color=True, show=False)

            print('dbg: PSNR\t', round(pc_error_results['  c[0],PSNRF'], 2),
                  '\t', round(pc_error_results['  c[1],PSNRF'], 2),
                  round(pc_error_results['  c[2],PSNRF'], 2))
            for k, v in pc_error_results.items(): results_one[k] = v
            if idx_rate == 0:
                results_list = [results_one]
            else:
                results_list.append(results_one)
        # merge all rate
        results = {'filename': os.path.split(filename)[-1]}
        one_keys = ['mseF,PSNR (p2point)','mseF,PSNR (p2plane)']
        multi_keys = ['bpp_att', 'bpp_hyper', 'enc_time', 'dec_time', '  c[0],PSNRF', '  c[1],PSNRF', '  c[2],PSNRF', 'pcqm', '1-pcqm']
        for idx_rate, results_one in enumerate(results_list):
            for k, v in results_one.items():
                if k in one_keys and idx_rate == 0: results[k] = v
                if k in multi_keys: results['R' + str(idx_rate) + '_' + k] = v
        results = pd.DataFrame([results])
        # merge all file
        if idx_file == 0:
            results_allfile = results.copy(deep=True)
        else:
            results_allfile = pd.concat([results_allfile, results], ignore_index=True)
        csvfile = os.path.join(output_rootdir, output_rootdir.split('/')[-2] + '.csv')
        results_allfile.to_csv(csvfile, index=False)
        print('save results to ', csvfile)

    return results_allfile


if __name__ == '__main__':
    """
    python test.py --mode='hyper' --output_rootdir='./output/hyper/'
    python test.py --mode='context'
    python test.py --mode='hyper'
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", default='guided')
    parser.add_argument("--input_rootdir", default='')
    parser.add_argument("--lossyGeo_rootdir", default='')
    parser.add_argument("--input_format", default='ply')
    parser.add_argument("--quick", type=bool, default=True, help="No coding, only estimate bitrate.")
    parser.add_argument("--resolution", type=int, default=1023)
    parser.add_argument("--frame", type=int, default=1)
    args = parser.parse_args()
    mode = 'test'
    args.output_rootdir = 'latest_output/' + mode + '/'
    os.makedirs(args.output_rootdir, exist_ok=True)
    print('dbg:\t output_rootdir:\t', args.output_rootdir)
    copyfile('test_lossyG_lossyA.py', args.output_rootdir + 'test_lossyG_lossyA.py')
    copyfile('model_pcac.py', args.output_rootdir + 'model_pcac.py')
    copyfile('models/basic_module.py', args.output_rootdir + 'basic_module.py')

    if args.mode == 'guided':
        from model_pcac import PCACModelContextHyperGuided
        ckptsdir_list = ['./ckpts/r04.pth']

        model_list_init = [PCACModelContextHyperGuided().to(device) for i in range(len(ckptsdir_list))]

    ###############################################################
    model_list = []
    for model, ckptsdir in zip(model_list_init, ckptsdir_list):
        ckpt = torch.load(ckptsdir, map_location='cuda:0')
        model.load_state_dict(ckpt['model'])
        model_list.append(model)
    results = test_compression(model_list, args.input_rootdir, args.output_rootdir, args.input_format, args.quick, args.lossyGeo_rootdir)
    print(results)

