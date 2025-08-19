import open3d as o3d
import argparse
import os
import numpy as np

def setup_headless_rendering():
    """
    Setup Open3D for headless rendering on Ubuntu VMs.
    This function sets the appropriate environment variables before Open3D operations.
    """
    # Set environment variables for headless CPU rendering
    # These must be set before any Open3D visualization operations
    os.environ['EGL_PLATFORM'] = 'surfaceless'  
    os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'  
    os.environ['GALLIUM_DRIVER'] = 'llvmpipe'
    
    # Set up virtual display environment
    os.environ['DISPLAY'] = ':1'
    
def ply_to_top_down_png_headless(ply_file, output_dir, img_size=(1920, 1080), bg_color=(1, 1, 1)):
    """
    Converts a .ply file to a top-down 2D PNG image using headless rendering.
    This version uses Open3D's OffscreenRenderer for headless environments.

    Args:
        ply_file (str): Path to the input .ply file.
        output_dir (str): Directory to save the output PNG file.
        img_size (tuple): A tuple containing the width and height of the output image.
        bg_color (tuple): A tuple representing the RGB background color (0-1 range).
    """
    # Setup headless rendering environment
    setup_headless_rendering()
    
    # Load the .ply file
    try:
        mesh = o3d.io.read_triangle_mesh(ply_file)
        if not mesh.has_triangles():
            # If the file doesn't contain a mesh, try reading it as a point cloud
            pcd = o3d.io.read_point_cloud(ply_file)
            if not pcd.has_points():
                print(f"Error: Cannot read {ply_file} as a mesh or point cloud.")
                return
            # Convert point cloud to mesh for rendering
            # Estimate normals first
            pcd.estimate_normals()
            # Use Poisson reconstruction to create mesh from point cloud
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
            if len(mesh.triangles) == 0:
                print(f"Warning: Could not create mesh from point cloud, using point cloud directly")
                geometry = pcd
            else:
                geometry = mesh
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
            else:
                # Add default colors if none exist
                mesh.paint_uniform_color([0.7, 0.7, 0.7])
            geometry = mesh
    except Exception as e:
        print(f"Error reading {ply_file}: {e}")
        return

    try:
        # Use OffscreenRenderer for headless rendering
        renderer = o3d.visualization.rendering.OffscreenRenderer(img_size[0], img_size[1])
        
        # Set up the scene
        renderer.scene.set_background(bg_color + (1.0,))  # Add alpha channel
        
        # Add geometry to scene
        if isinstance(geometry, o3d.geometry.PointCloud):
            # For point clouds, use a simple material
            material = o3d.visualization.rendering.MaterialRecord()
            material.shader = "defaultUnlit"
            material.point_size = 5.0
            renderer.scene.add_geometry("pointcloud", geometry, material)
        else:
            # For meshes, use default lit material
            material = o3d.visualization.rendering.MaterialRecord()
            material.shader = "defaultLit"
            renderer.scene.add_geometry("mesh", geometry, material)
        
        # Get geometry bounds for camera positioning
        bbox = geometry.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = bbox.get_extent()
        
        # Set up camera for top-down view
        # Calculate camera distance based on geometry size
        max_extent = np.max(extent)
        camera_distance = max_extent
        
        # Position camera above the geometry looking down
        camera_pos = center + np.array([0, 0, camera_distance])
        
        # Set up the camera to match fallback method's view
        renderer.setup_camera(60, center, camera_pos, [0, 1, 0])
        
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate output filename
        base_name = os.path.basename(ply_file)
        file_name = os.path.splitext(base_name)[0]
        output_path = os.path.join(output_dir, f"{file_name}_top_down.png")
        
        # Render the image
        image = renderer.render_to_image()
        
        # Save the image
        o3d.io.write_image(output_path, image)
        
        print(f"Successfully saved top-down image to {output_path}")
        
    except Exception as e:
        print(f"Error during headless rendering: {e}")
        print("Falling back to alternative rendering method...")
        # Fallback to the original method with additional environment setup
        return ply_to_top_down_png_fallback(ply_file, output_dir, img_size, bg_color)

def ply_to_top_down_png_fallback(ply_file, output_dir, img_size=(1920, 1080), bg_color=(1, 1, 1)):
    """
    Fallback method using traditional Open3D Visualizer with enhanced headless setup.
    """
    # Additional environment setup for fallback
    os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
    os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'
    
    try:
        # Load the geometry
        mesh = o3d.io.read_triangle_mesh(ply_file)
        if not mesh.has_triangles():
            pcd = o3d.io.read_point_cloud(ply_file)
            if not pcd.has_points():
                print(f"Error: Cannot read {ply_file} as a mesh or point cloud.")
                return
            pcd.estimate_normals()
            geometry = pcd
        else:
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
            if mesh.has_vertex_colors():
                colors = np.asarray(mesh.vertex_colors)
                colors = np.clip(colors, 0, 1)
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            else:
                mesh.paint_uniform_color([0.7, 0.7, 0.7])
            geometry = mesh

        # Create a visualizer with headless configuration
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=img_size[0], height=img_size[1], visible=False)
        vis.add_geometry(geometry)

        # Get geometry center and scale for better view
        bbox = geometry.get_axis_aligned_bounding_box()
        center = bbox.get_center()

        # Set the view to top-down
        ctr = vis.get_view_control()
        ctr.set_lookat(center)
        ctr.set_front([0, 0, 1])  # Top-down view (Z-up)
        #ctr.set_up([0, 1, 0])     # Y-up orientation
        
        # Configure render settings for headless environment
        opt = vis.get_render_option()
        opt.background_color = np.asarray(bg_color)
        opt.point_size = 5.0
        opt.mesh_show_back_face = True
        opt.light_on = False

        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the image
        base_name = os.path.basename(ply_file)
        file_name = os.path.splitext(base_name)[0]
        output_path = os.path.join(output_dir, f"{file_name}_top_down.png")

        vis.capture_screen_image(output_path, do_render=True)
        vis.destroy_window()

        print(f"Successfully saved top-down image to {output_path} (using fallback method)")

    except Exception as e:
        print(f"Error in fallback rendering: {e}")
        print("Both rendering methods failed. Please check your Open3D installation and system dependencies.")

def ply_to_top_down_png(ply_file, output_dir, img_size=(1920, 1080), bg_color=(1, 1, 1)):
    """
    Main function that tries headless rendering first, then falls back if needed.
    """
    # Try headless rendering first
    try:
        ply_to_top_down_png_headless(ply_file, output_dir, img_size, bg_color)
    except Exception as e:
        print(f"Headless rendering failed: {e}")
        print("Attempting fallback method...")
        ply_to_top_down_png_fallback(ply_file, output_dir, img_size, bg_color)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a .ply file to a top-down 2D PNG image with headless rendering support.")
    parser.add_argument("ply_file", type=str, help="Path to the input .ply file.")
    parser.add_argument("--output_dir", type=str, default="output_images", help="Directory to save the output PNG file.")
    parser.add_argument("--img_width", type=int, default=1920, help="Width of the output image.")
    parser.add_argument("--img_height", type=int, default=1080, help="Height of the output image.")
    parser.add_argument("--bg_color", type=float, nargs=3, default=[1, 1, 1], help="Background color (R G B), values from 0 to 1.")
    parser.add_argument("--method", type=str, choices=['auto', 'headless', 'fallback'], default='auto', 
                      help="Rendering method: auto (try headless first), headless (OffscreenRenderer only), fallback (traditional Visualizer)")

    args = parser.parse_args()

    if args.method == 'headless':
        ply_to_top_down_png_headless(
            args.ply_file,
            args.output_dir,
            img_size=(args.img_width, args.img_height),
            bg_color=tuple(args.bg_color)
        )
    elif args.method == 'fallback':
        ply_to_top_down_png_fallback(
            args.ply_file,
            args.output_dir,
            img_size=(args.img_width, args.img_height),
            bg_color=tuple(args.bg_color)
        )
    else:  # auto
        ply_to_top_down_png(
            args.ply_file,
            args.output_dir,
            img_size=(args.img_width, args.img_height),
            bg_color=tuple(args.bg_color)
        ) 