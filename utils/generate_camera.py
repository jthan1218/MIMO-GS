import numpy as np
import math

from scene.cameras import Camera


def generate_new_cam(r_d, tx, image_height=90, image_width=360):
    """Create a virtual camera with configurable output resolution."""

    # rot = Rotation.from_rotvec(r_d).as_matrix()
    rot = r_d
    trans = tx

    fovx = np.deg2rad(180)
    fovy = np.deg2rad(180)

    cam = Camera(
        R=rot,
        colmap_id=None,
        T=trans,
        FoVx=fovx,
        FoVy=fovy,
        image=None,
        image_name=None,
        uid=None,
        invdepthmap=None,
        depth_params=None,
    )
    cam.image_width = image_width
    cam.image_height = image_height

    return cam


