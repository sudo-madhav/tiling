"""Shared pieces for the WCSI ablation harness. See PROTOCOL.md.

Nothing here reads the network or writes into the image directories.
"""

import json
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".ppm"}

TERMS = ("E_RG", "E_GJ", "E_PCP", "E_cyto", "E_sheaf")

# Primary methods. Anything else in the config is an optional re-inference run.
PRIMARY_METHODS = ("bicubic", "vosr", "casr", "seedvr")


def load_config(path):
    path = Path(path)
    with path.open() as fh:
        cfg = json.load(fh)
    if "gt" not in cfg or "scales" not in cfg or "out" not in cfg:
        raise ValueError(f"{path} needs keys gt, out, scales")
    cfg["_config_path"] = str(path)
    return cfg


def method_spec(value):
    """A method entry is either a directory string or {path, model_tile_sr}.

    An empty path means the run is not on disk. For the method name bicubic an empty
    path means synthesize the upsampling inside the scorer.
    """
    if isinstance(value, str):
        return {"path": value, "model_tile_sr": None}
    if not isinstance(value, dict) or "path" not in value:
        raise ValueError(f"method entry must be a path or {{\"path\": ...}}, got {value!r}")
    return {"path": value["path"], "model_tile_sr": value.get("model_tile_sr")}


def index_images(directory):
    """stem -> path. Raises if two files share a stem."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    out = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        if path.stem in out:
            raise ValueError(f"two files share the stem {path.stem!r} in {directory}")
        out[path.stem] = path
    return out


def header_size(path):
    """(width, height) from the file header, without decoding the pixels."""
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def sr_pixels_to_lr(value, scale, name):
    if value % scale != 0:
        raise ValueError(f"{name}={value} SR px is not divisible by scale {scale}")
    return value // scale


def analysis_grid(scale, tile_sr, overlap_sr, offset_sr):
    """Convert an analysis grid given in upscaled pixels into LR pixels.

    WCSI's tile and overlap arguments are in LR pixels. The experiments quote tile
    sizes in the upscaled image (256, 512, 1024), which is also the space VOSR's
    --tile_size and SeedVR's VAE tile use.
    """
    tile = sr_pixels_to_lr(tile_sr, scale, "tile")
    overlap = sr_pixels_to_lr(overlap_sr, scale, "overlap")
    offset = sr_pixels_to_lr(offset_sr, scale, "offset")
    if tile < 8:
        raise ValueError(f"tile {tile_sr} SR px is {tile} LR px at {scale}x; WCSI needs at least 8")
    if not 0 <= overlap < tile:
        raise ValueError(f"overlap {overlap_sr} SR px must be in [0, {tile_sr}) at {scale}x")
    if offset < 0:
        raise ValueError("offset must be >= 0")
    return tile, overlap, offset


def default_overlap_sr(tile_sr):
    """WCSI's own default, tile/16, expressed in SR pixels. Stays divisible by 2, 4 and 8."""
    return tile_sr // 16


def lattice_coords(height, width, tile, overlap, offset):
    """Regular tile rectangles (dicts with x, y, w, h) in LR pixels, starting at offset.

    Unlike wcsi.tile_starts, this does not pull the last tile flush with the border.
    The flush tile would put a seam back at the image edge and defeat an offset probe.
    """
    step = tile - overlap
    ys = list(range(offset, height - tile + 1, step))
    xs = list(range(offset, width - tile + 1, step))
    if not ys or not xs:
        raise ValueError(f"offset grid is empty for a {width}x{height} LR image, "
                         f"tile {tile}, overlap {overlap}, offset {offset}")
    return [dict(x=x, y=y, w=tile, h=tile) for y in ys for x in xs]


def usable_lr_size(lr_wh, sr_wh, scale):
    """The LR size WCSI will actually score, after its integer-scale crop."""
    lr_w, lr_h = lr_wh
    sr_w, sr_h = sr_wh
    if scale < 2 or abs(sr_h - scale * lr_h) >= scale or abs(sr_w - scale * lr_w) >= scale:
        raise ValueError(f"SR {sr_w}x{sr_h} is not an integer multiple (>= 2) of LR {lr_w}x{lr_h} at {scale}x")
    return min(lr_w, sr_w // scale), min(lr_h, sr_h // scale)


def result_path(out_dir, dataset, scale, method, tile_sr, overlap_sr, offset_sr, stem):
    folder = (Path(out_dir) / dataset / f"x{scale}" / method
              / f"tile{tile_sr}_ov{overlap_sr}_off{offset_sr}")
    return folder / f"{stem}.json"


def select_stems(stems, limit=0, shard=None):
    stems = sorted(stems)
    if shard is not None:
        index, parts = shard
        if parts < 1 or not 0 <= index < parts:
            raise ValueError(f"shard must be i/n with 0 <= i < n, got {index}/{parts}")
        stems = [stem for k, stem in enumerate(stems) if k % parts == index]
    if limit:
        stems = stems[:limit]
    return stems


def parse_shard(text):
    if text is None:
        return None
    index, parts = text.split("/")
    return int(index), int(parts)


def parse_int_list(text):
    return [int(v) for v in text.split(",") if v.strip()]


def parse_overlaps(text, tiles):
    """'tile/16' or a comma list of SR-pixel overlaps, applied to every tile."""
    if text.strip() == "tile/16":
        return {tile: [default_overlap_sr(tile)] for tile in tiles}
    overlaps = parse_int_list(text)
    return {tile: overlaps for tile in tiles}


def parse_offsets(text, tiles):
    """'0', '0,half', or a comma list. 'half' is tile/2 for that tile."""
    out = {}
    tokens = [v.strip() for v in text.split(",") if v.strip()]
    for tile in tiles:
        values = []
        for token in tokens:
            if token == "half":
                values.append(tile // 2)
            else:
                values.append(int(token))
        out[tile] = values
    return out
