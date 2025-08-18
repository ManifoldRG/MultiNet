import open3d as o3d
import argparse
import os
import numpy as np

def ply_to_top_down_png(ply_file, output_dir, img_size=(1920, 1080), bg_color=(0, 0, 0, 0)):
    """
    Converts a .ply file to a top-down 2D PNG image.

    Args:
        ply_file (str): Path to the input .ply file.
        output_dir (str): Directory to save the output PNG file.
        img_size (tuple): A tuple containing the width and height of the output image.
        bg_color (tuple): A tuple representing the RGBA background color.
    """
    # Load the .ply file
    try:
        mesh = o3d.io.read_triangle_mesh(ply_file)
        if not mesh.has_triangles():
            # If the file doesn't contain a mesh, try reading it as a point cloud
            pcd = o3d.io.read_point_cloud(ply_file)
            if not pcd.has_points():
                print(f"Error: Cannot read {ply_file} as a mesh or point cloud.")
                return
            # Preprocess point cloud
            pcd.estimate_normals()
            geometry = pcd
        else:
            # Preprocess mesh
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
            # Ensure proper color handling
            if mesh.has_vertex_colors():
                colors = np.asarray(mesh.vertex_colors)
                # Normalize colors to ensure proper range
                colors = np.clip(colors, 0, 1)
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            geometry = mesh
    except Exception as e:
        print(f"Error reading {ply_file}: {e}")
        return

    # Create a visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=img_size[0], height=img_size[1], visible=False)
    vis.add_geometry(geometry)

    # Get geometry center and scale for better view
    bbox = geometry.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    scale = np.linalg.norm(bbox.get_extent())

    # Set the view to top-down with adjusted position
    ctr = vis.get_view_control()
    ctr.set_lookat(center)
    ctr.set_front([0, 0, 1])  # Top-down view (Z-up)
    #ctr.set_up([0, -1, 0])
    
    # Configure render settings
    opt = vis.get_render_option()
    opt.background_color = np.asarray(bg_color[:3])  # Only use RGB components
    opt.point_size = 5.0  # Slightly smaller points
    opt.mesh_show_back_face = True
    opt.light_on = True
    opt.point_show_normal = True  # Re-enable normals but keep other settings modest

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the image
    base_name = os.path.basename(ply_file)
    file_name = os.path.splitext(base_name)[0]
    output_path = os.path.join(output_dir, f"{file_name}_top_down.png")

    
    vis.capture_screen_image(output_path, do_render=True)

    # Clean up
    vis.destroy_window()

    print(f"Successfully saved top-down image to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a .ply file to a top-down 2D PNG image.")
    parser.add_argument("ply_file", type=str, help="Path to the input .ply file.")
    parser.add_argument("--output_dir", type=str, default="output_images", help="Directory to save the output PNG file.")
    parser.add_argument("--img_width", type=int, default=1920, help="Width of the output image.")
    parser.add_argument("--img_height", type=int, default=1080, help="Height of the output image.")
    parser.add_argument("--bg_color", type=float, nargs=3, default=[1, 1, 1], help="Background color (R G B), values from 0 to 1.")

    args = parser.parse_args()

    ply_to_top_down_png(
        args.ply_file,
        args.output_dir,
        img_size=(args.img_width, args.img_height),
        bg_color=tuple(args.bg_color)
    ) 