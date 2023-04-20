#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import datetime
import sys
import argparse
import os
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
import commentjson as json

import numpy as np

import shutil
import time

from common import *
from scenes import *

from tqdm import tqdm
from omegaconf import OmegaConf

pyngp_path = opj(opd(__file__), "build")
sys.path.append(pyngp_path)
import pyngp as ngp
# from vis import plot_density_grid

from ros_communication import listener        

def test():
    print("Evaluating test transforms from ", opt.test_transforms)
    with open(opt.test_transforms) as f:
        test_transforms = json.load(f)
    data_dir = os.path.dirname(opt.test_transforms)
    totmse = 0
    totpsnr = 0
    totssim = 0
    totcount = 0
    minpsnr = 1000
    maxpsnr = 0

    # Evaluate metrics on black background
    testbed.background_color = [0.0, 0.0, 0.0, 1.0]

    # Prior nerf papers don't typically do multi-sample anti aliasing.
    # So snap all pixels to the pixel centers.
    testbed.snap_to_pixel_centers = True
    spp = 8

    testbed.nerf.rendering_min_transmittance = 1e-4

    testbed.fov_axis = 0
    testbed.fov = test_transforms["camera_angle_x"] * 180 / np.pi
    testbed.shall_train = False

    with tqdm(
        list(enumerate(test_transforms["frames"])),
        unit="images",
        desc=f"Rendering test frame",
    ) as t:
        for i, frame in t:
            p = frame["file_path"]
            if "." not in p:
                p = p + ".png"
            ref_fname = os.path.join(data_dir, p)
            if not os.path.isfile(ref_fname):
                ref_fname = os.path.join(data_dir, p + ".png")
                if not os.path.isfile(ref_fname):
                    ref_fname = os.path.join(data_dir, p + ".jpg")
                    if not os.path.isfile(ref_fname):
                        ref_fname = os.path.join(data_dir, p + ".jpeg")
                        if not os.path.isfile(ref_fname):
                            ref_fname = os.path.join(data_dir, p + ".exr")

            ref_image = read_image(ref_fname)

            # NeRF blends with background colors in sRGB space, rather than first
            # transforming to linear space, blending there, and then converting back.
            # (See e.g. the PNG spec for more information on how the `alpha` channel
            # is always a linear quantity.)
            # The following lines of code reproduce NeRF's behavior (if enabled in
            # testbed) in order to make the numbers comparable.
            if testbed.color_space == ngp.ColorSpace.SRGB and ref_image.shape[2] == 4:
                # Since sRGB conversion is non-linear, alpha must be factored out of it
                ref_image[..., :3] = np.divide(
                    ref_image[..., :3],
                    ref_image[..., 3:4],
                    out=np.zeros_like(ref_image[..., :3]),
                    where=ref_image[..., 3:4] != 0,
                )
                ref_image[..., :3] = linear_to_srgb(ref_image[..., :3])
                ref_image[..., :3] *= ref_image[..., 3:4]
                ref_image += (1.0 - ref_image[..., 3:4]) * testbed.background_color
                ref_image[..., :3] = srgb_to_linear(ref_image[..., :3])

            if i == 0:
                write_image("ref.png", ref_image)

            testbed.set_nerf_camera_matrix(np.matrix(frame["transform_matrix"])[:-1, :])
            image = testbed.render(ref_image.shape[1], ref_image.shape[0], spp, True)

            if i == 0:
                write_image("out.png", image)

            diffimg = np.absolute(image - ref_image)
            diffimg[..., 3:4] = 1.0
            if i == 0:
                write_image("diff.png", diffimg)

            A = np.clip(linear_to_srgb(image[..., :3]), 0.0, 1.0)
            R = np.clip(linear_to_srgb(ref_image[..., :3]), 0.0, 1.0)
            mse = float(compute_error("MSE", A, R))
            ssim = float(compute_error("SSIM", A, R))
            totssim += ssim
            totmse += mse
            psnr = mse2psnr(mse)
            totpsnr += psnr
            minpsnr = psnr if psnr < minpsnr else minpsnr
            maxpsnr = psnr if psnr > maxpsnr else maxpsnr
            totcount = totcount + 1
            t.set_postfix(psnr=totpsnr / (totcount or 1))

    psnr_avgmse = mse2psnr(totmse / (totcount or 1))
    psnr = totpsnr / (totcount or 1)
    ssim = totssim / (totcount or 1)
    print(f"PSNR={psnr} [min={minpsnr} max={maxpsnr}] SSIM={ssim}")


def screenshot():
    testbed.autofocus = True
    print("Screenshot transforms from ", opt.screenshot_transforms)
    with open(opt.screenshot_transforms) as f:
        ref_transforms = json.load(f)
    screenshot_dir = opj(output_dir, "screenshot")
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir)
    # testbed.fov_axis = 0
    # testbed.fov = ref_transforms["camera_angle_x"] * 180 / np.pi
    # render_seconds_last = 0
    for render_mode in [ngp.RenderMode.Shade,ngp.RenderMode.Positions,ngp.RenderMode.Depth]:
        testbed.render_mode=render_mode
        for idx in range(len(ref_transforms["frames"])):
            f = ref_transforms["frames"][int(idx)]
            cam_matrix = f["transform_matrix"]
            testbed.set_nerf_camera_matrix(np.matrix(cam_matrix)[:-1, :])
            outname = os.path.join(screenshot_dir, os.path.basename(f["file_path"]).split(".")[0]+"_"+render_mode.name)

            # Some NeRF datasets lack the .png suffix in the dataset metadata
            if not os.path.splitext(outname)[1]:
                outname = outname + ".png"

            print(f"rendering {outname}")
            # tic = time.time()
            image = testbed.render(
                opt.width or int(ref_transforms["w"]),
                opt.height or int(ref_transforms["h"]),
                opt.screenshot_spp,
                True,
            )
            # toc = time.time()
            # render_seconds_last += toc - tic
            write_image(outname, image)
    # with open(log_path, "a") as f:
    #     f.write(
    #         "seconds per render:{}\n".format(
    #             render_seconds_last / len(ref_transforms["frames"])
    #         )
    #     )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ngp exp")
    parser.add_argument(
        "-p",
        type=str,
        default=opj(opd(opd(__file__)), "configs", "default.yaml"),
        help="yaml file path",
    )
    parser.add_argument("-g", type=str, default="0")
    args = parser.parse_args()
    output_dir = opj(
        opd(opd(__file__)),
        "outputs",
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = args.g
    os.makedirs(output_dir)

    # shutil.copyfile("utils/transforms.py", opj(output_dir, "transform.py"))
    # shutil.copyfile("utils/transforms_new.py", opj(output_dir, "transform_new.py"))

    log_path = opj(output_dir, "profile.log")
    opt = OmegaConf.load(args.p)
    OmegaConf.save(opt, opj(output_dir, "config.yaml"))
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    testbed.nerf.training.depth_optimize_ray_tracing = opt["depth_optimize_ray_tracing"]
    testbed.nerf.training.depth_optimize_density_grid = opt["depth_optimize_density_grid"]
    # testbed.enable_depth_loading = opt["enable_depth_loading"]
    # if testbed.depth_optimize_density_grid or testbed.depth_optimize_ray_tracing:
    #     assert testbed.enable_depth_loading
    testbed.nerf.sharpen = opt.sharpen
    testbed.exposure = opt.exposure
    testbed.nerf.training.minimum_thickness_of_opaque_object_realunit = opt.minimum_thickness_of_opaque_object_realunit
    testbed.nerf.training.depth_supervision_lambda = opt.depth_supervision_lambda
    testbed.nerf.training.max_empty_samples_per_ray = opt.max_empty_samples_per_ray
    testbed.nerf.training.lower_limit_opaque_point_weight = (
        opt.lower_limit_opaque_point_weight
    )
    testbed.nerf.training.max_samples_behind_surface = opt.max_samples_behind_surface
    testbed.nerf.training.empty_density_loss_scale = opt.empty_density_loss_scale
    testbed.nerf.training.opaque_density_loss_scale = opt.opaque_density_loss_scale

    # listener(testbed)

    testbed.load_training_data(opt.training_data)
    # sys.exit(0)
    if opt.load_snapshot:
        snapshot = opt.load_snapshot
        print("Loading snapshot ", snapshot)
        testbed.load_snapshot(snapshot)
    else:
        network_config = OmegaConf.to_container(opt.network)
        network_config_path = opj(output_dir, "network.json")
        json.dump(network_config, open(network_config_path, "w"), indent=2)
        testbed.reload_network_from_file(network_config_path)

    # print(f"images_for_training:{testbed.nerf.training.n_images_for_training}")
    while (testbed.nerf.training.n_images_for_training == 0):
        continue

    testbed.shall_train = True
    testbed.nerf.render_with_camera_distortion = True
    old_training_step = 0
    n_steps = opt.n_steps
    train_seconds = opt.train_seconds
    tqdm_last_update = 0
    train_seconds_last = 0
    # TODO:
    # testbed.change_rays_per_batch(opt.rays_per_batch)
    if n_steps > 0:
        with tqdm(desc="Training", total=n_steps, unit="step") as t:
            while True:
                if (
                    testbed.training_step >= n_steps
                    or train_seconds_last >= train_seconds
                ):
                    with open(log_path, "a") as f:
                        f.write(
                            "train steps:{}\ntrain seconds:{}\n".format(
                                testbed.training_step, train_seconds_last
                            )
                        )
                        f.write(
                            "seconds per train step:{}\n".format(
                                train_seconds_last / testbed.training_step
                            )
                        )
                    break
                tic = time.time()
                testbed.frame()
                # print(f"training_step:{testbed.training_step}")
                # print(f"images_for_training:{testbed.nerf.training.n_images_for_training}")
                toc = time.time()
                train_seconds_last += toc - tic
                # Update progress bar
                if testbed.training_step < old_training_step or old_training_step == 0:
                    old_training_step = 0
                    t.reset()
                now = time.monotonic()
                if now - tqdm_last_update > 0.1:
                    t.update(testbed.training_step - old_training_step)
                    t.set_postfix(loss=testbed.loss)
                    old_training_step = testbed.training_step
                    tqdm_last_update = now

    if opt.save_snapshot:
        print("Saving snapshot ", opt.save_snapshot)
        testbed.save_snapshot(opt.save_snapshot, False)

    if opt.test_transforms:
        test()

    if opt.save_mesh:
        res = opt.marching_cubes_res or 256
        print(
            f"Generating mesh via marching cubes and saving to {opt.save_mesh}. Resolution=[{res},{res},{res}]"
        )
        testbed.compute_and_save_marching_cubes_mesh(opt.save_mesh, [res, res, res])

    if opt.screenshot:
        screenshot_dir = opj(output_dir, "screenshot")
        # os.makedirs(screenshot_dir)
        outname = os.path.join(screenshot_dir, "render")
        print(f"Rendering {outname}.png")
        image = testbed.render(
                1920,
                1080,
                opt.screenshot_spp,
                True,
            )
        if os.path.dirname(outname) != "":
            os.makedirs(os.path.dirname(outname), exist_ok=True)
        write_image(outname + ".png", image)
    if opt.screenshot_transforms:
        screenshot()
    # density_grid_dict = testbed.get_density_grid()
    # camera_pos_dir_dict = testbed.get_camera_pos_dir()
    # plot_density_grid(
    #     density_grid_dict,
    #     camera_pos_dir_dict,
    #     output_dir,
    #     inverse_z=True,
    #     show_camera=True,
    #     show_density_grid=False,
    #     show_init_empty_cells=False,
    #     show_init_opaque_cells=True,
    #     show_init_unkown_cells=False,
    # )
    # plot_density_grid(
    #     density_grid_dict,
    #     camera_pos_dir_dict,
    #     output_dir,
    #     inverse_z=True,
    #     show_camera=True,
    #     show_density_grid=True,
    #     show_init_empty_cells=False,
    #     show_init_opaque_cells=False,
    #     show_init_unkown_cells=False,
    # )
    # testbed.compute_and_save_png_slices(filename=opj(output_dir, 'depth.png'),
    #                                     resolution=256)