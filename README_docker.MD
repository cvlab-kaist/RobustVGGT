# Docker Usage

This guide assumes the image is built with the tag:

```bash
robust-vggt-gh200
```

This image runs `robust_vggt.py` with:

```bash
python robust_vggt.py --image-dir examples/trevi
```

by default.

## Build The Image

Build from the repository root with:

```bash
docker build -t robust-vggt-gh200 -f Dockerfile .
```

After the build completes, you can verify the image exists with:

```bash
docker images | rg robust-vggt-gh200
```

## Where Results Are Saved

The script writes outputs to the directory given by `--exp-name`.

If you do not pass `--exp-name`, the default is:

```bash
demo_result
```

Inside the container, that means the results are saved to:

```bash
/workspace/RobustVGGT/demo_result
```

## Can You See Results Outside The Container?

Yes, but only if you mount a host directory into the container.

- If you run the container without a volume mount, results stay inside the container filesystem.
- If you also use `--rm`, the container is deleted after exit, so those results are lost.

## Recommended Run Command

Mount the project directory from the host into the container so outputs are written back to the host:

```bash
docker run --rm --gpus all \
  -v /home/ubuntu/RobustVGGT:/workspace/RobustVGGT \
  -it robust-vggt-gh200 \
  robust_vggt.py \
  --image-dir examples/stable-users-699100cd4505a30060749938-veo-videos-fe36dbd306a786cad7a7a45b9367b81d-1775190663 \
  --exp-name output/stable-users-699100cd4505a30060749938-veo-videos-fe36dbd306a786cad7a7a45b9367b81d-1775190663```

With this command, the output will be visible on the host at:

```bash
/home/ubuntu/RobustVGGT/output/stable-users-699100cd4505a30060749938-veo-videos-fe36dbd306a786cad7a7a45b9367b81d-1775190663
```

(That path matches the `--exp-name` value above; change it if you use a different `--exp-name`.)

## Visualize Saved Results

The Docker image also includes `visualize_demo_result.py`, which can render:

- RGB contact sheets
- depth visualizations
- confidence visualizations
- a top-down camera trajectory image
- a sparse `.ply` point cloud

To visualize a result saved at `output/demo_result` on the host:

```bash
docker run --rm -it \
  -v /home/ubuntu/RobustVGGT:/workspace/RobustVGGT \
  --entrypoint python \
  robust-vggt-gh200 \
  /workspace/RobustVGGT/visualize_demo_result.py --input /workspace/RobustVGGT/output/demo_result
```

The generated files will be written to:

```bash
/home/ubuntu/RobustVGGT/output/demo_result/visualizations
```

If you want to point to a specific `.npz` file instead of a directory, run:

```bash
docker run --rm -it \
  -v /home/ubuntu/RobustVGGT:/workspace/RobustVGGT \
  --entrypoint python \
  robust-vggt-gh200 \
  /workspace/RobustVGGT/visualize_demo_result.py \
  --input /workspace/RobustVGGT/output/demo_result/predictions_survived.npz
```

If your result directory is `demo_result` in the repository root instead, run:

```bash
docker run --rm -it \
  -v /home/ubuntu/RobustVGGT:/workspace/RobustVGGT \
  --entrypoint python \
  robust-vggt-gh200 \
  /workspace/RobustVGGT/visualize_demo_result.py --input /workspace/RobustVGGT/demo_result
```

## Why `--entrypoint` Is Needed For Visualization

The image is configured to run inference by default:

```bash
ENTRYPOINT ["python", "robust_vggt.py"]
CMD ["--image-dir", "examples/trevi"]
```

So when you want to run the visualization script instead, override the entrypoint and call Python explicitly:

```bash
docker run ... --entrypoint python robust-vggt-gh200 visualize_demo_result.py ...
```

## Common Issues

- If outputs do not appear on the host, check that you mounted the host directory with `-v`.
- If you use `--rm` without mounting outputs, generated files are deleted with the container.
- If `--gpus all` fails, confirm that Docker can access the NVIDIA runtime on the host.
- If you only want visualization of existing `.npz` files, you do not need `--gpus all`.
- If you saved results with the default `--exp-name`, the directory is `demo_result`, not `output/demo_result`.

## Notes

- The container working directory is `/workspace/RobustVGGT`.
- `--exp-name` can be a relative path such as `output/demo_result`.
- You can replace `examples/trevi` with your own mounted image directory.
- The image contains an internal virtual environment at `/opt/venv`, and `PATH` is already configured to use it.
