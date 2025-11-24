import bpy
import bpy_extras
import os
import random
import math
import json
from mathutils import Vector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "source_files/models/")
BACKGROUND_DIR = os.path.join(BASE_DIR, "source_files/background/")
FOREGROUND_DIR = os.path.join(BASE_DIR, "source_files/foreground/")

OUT_IMAGE = os.path.join(BASE_DIR, "datasets/images_with_foreground/")
OUT_JSON = os.path.join(BASE_DIR, "datasets/annotations_with_foreground.json")

NUM_IMAGES = 2520  # total number of images to generate
IMAGES_PER_LOOP = 20  # number of images per loop
DECENTER_CHANCE = 1  # chance to decenter the camera
FOREGROUND_CHANCE = 0.4  # chance to add foreground
IMAGE_RES = (640, 640)  # resolution of the output images

counter = 1  # Initialize a global counter for COCO annotations

# Find all DAE files in the MODEL_DIR
def find_models():
    all_models = []
    for folder in os.listdir(MODEL_DIR):
        model_paths = []
        folder_path = os.path.join(MODEL_DIR, folder)
        if os.path.isdir(folder_path):
            dae_files = [f for f in os.listdir(folder_path) if f.endswith('.dae')]
            for dae_file in dae_files:
                model_path = os.path.join(folder_path, dae_file)
                model_paths.append(model_path)
            all_models.append((folder, model_paths))
    return all_models

# Find all PNG background images
def find_background():
    backgrounds = []
    if os.path.exists(BACKGROUND_DIR):
        for file in os.listdir(BACKGROUND_DIR):
            if file.endswith('.png'):
                background = os.path.join(BACKGROUND_DIR, file)
                backgrounds.append((file, background))
    return backgrounds

# Find all PNG foreground images
def find_foreground():
    foregrounds = []
    if os.path.exists(FOREGROUND_DIR):
        for file in os.listdir(FOREGROUND_DIR):
            if file.endswith(('.png')):
                foreground_path = os.path.join(FOREGROUND_DIR, file)
                foregrounds.append((file, foreground_path))
    return foregrounds

# Clear the scene of all objects
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    return

# Import a model from file
def import_model(model_path):
    bpy.ops.wm.collada_import(filepath=model_path)
    objs = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    return objs[0] if objs else None

# Get the size and the center of the model
def get_model_info(obj):
    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    first_corner = bbox[0]
    min_x = max_x = first_corner.x
    min_y = max_y = first_corner.y
    min_z = max_z = first_corner.z

    for corner in bbox:
        min_x = min(min_x, corner.x)
        max_x = max(max_x, corner.x)
        min_y = min(min_y, corner.y)
        max_y = max(max_y, corner.y)
        min_z = min(min_z, corner.z)
        max_z = max(max_z, corner.z)

    size = max(max_x - min_x, max_y - min_y, max_z - min_z)
    center = Vector(((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2))
    return size, center

# Add a camera to the scene
def add_camera(model_size, model_center, image_index):

    # Random angle and radius for camera position
    angle = (image_index / IMAGES_PER_LOOP) * 2 * math.pi + random.uniform(-0.3, 0.3)
    radius = model_size * random.uniform(5.0, 30.0)

    x = radius * math.cos(angle)
    y = radius * math.sin(angle)
    z = model_size * random.uniform(-0.5, 3.0) # Random height

    # Random camera decentering
    center_offset = model_center.copy()
    if random.random() < DECENTER_CHANCE:
        offset_range = radius * 0.15
        center_offset.x += random.uniform(-offset_range, offset_range)
        center_offset.y += random.uniform(-offset_range, offset_range)
        center_offset.z += random.uniform(-offset_range, offset_range)

    bpy.ops.object.camera_add(location=(x, y, z))
    cam = bpy.context.active_object
    cam.data.clip_end = 10000
    cam.data.sensor_width = 25

    direction = center_offset - Vector((x, y, z))
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    bpy.context.scene.camera = cam
    return (x, y, z)

# Add a light source to the scene
def add_light(model_size, camera_location):
    for light in bpy.data.lights:
        bpy.data.lights.remove(light)

    light_data = bpy.data.lights.new(name="FillLight", type='POINT')
    light_obj = bpy.data.objects.new(name="FillLight", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)

    camera_x, camera_y, camera_z = camera_location

    # Random light intensity
    offset_distance = model_size * random.uniform(1.0, 2.0)
    angle_offset = random.uniform(-0.5, 0.5)

    x = camera_x + offset_distance * math.cos(angle_offset)
    y = camera_y + offset_distance * math.sin(angle_offset)
    z = camera_z + model_size * random.uniform(0.5, 1.5)

    light_obj.location = (x, y, z)
    light_data.energy = 500 * (model_size ** 2)
    return

# Add background
def add_background(background_path):
    world = bpy.context.scene.world
    world.use_nodes = True
    world.node_tree.nodes.clear()

    world_output = world.node_tree.nodes.new(type='ShaderNodeOutputWorld')
    background_shader = world.node_tree.nodes.new(type='ShaderNodeBackground')
    env_texture = world.node_tree.nodes.new(type='ShaderNodeTexEnvironment')
    mapping = world.node_tree.nodes.new(type='ShaderNodeMapping')
    tex_coord = world.node_tree.nodes.new(type='ShaderNodeTexCoord')

    bpy.ops.image.open(filepath=background_path)
    img = bpy.data.images[os.path.basename(background_path)]
    env_texture.image = img

    world.node_tree.links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
    world.node_tree.links.new(mapping.outputs['Vector'], env_texture.inputs['Vector'])
    world.node_tree.links.new(env_texture.outputs['Color'], background_shader.inputs['Color'])
    world.node_tree.links.new(background_shader.outputs['Background'], world_output.inputs['Surface'])

    # Random texture rotation
    rotation_z = random.uniform(0, 2 * math.pi)
    mapping.inputs['Rotation'].default_value = (0, 0, rotation_z)

    # Random strength for lighting variation
    strength = random.uniform(0.8, 1.5)
    background_shader.inputs['Strength'].default_value = strength

# Add foreground elements that partially hide the model
def add_foreground(model_size, model_center, camera_location, foreground_path):
    # Remove existing foreground planes
    for obj in bpy.context.scene.objects:
        if obj.name.startswith("Foreground"):
            bpy.data.objects.remove(obj, do_unlink=True)

    camera_x, camera_y, camera_z = camera_location
    model_x, model_y, model_z = model_center

    # Calculate foreground position and randomize its placement
    offset_range = model_size * 0.3
    fg_x = model_x + (camera_x - model_x) * 0.7 + random.uniform(-offset_range, offset_range)
    fg_y = model_y + (camera_y - model_y) * 0.7 + random.uniform(-offset_range, offset_range)
    fg_z = model_z + (camera_z - model_z) * 0.7 + random.uniform(-offset_range * 0.5, offset_range * 0.5)

    # Create plane and randomize its size
    plane_size = model_size * random.uniform(1.5, 3.0)
    bpy.ops.mesh.primitive_plane_add(size=plane_size, location=(fg_x, fg_y, fg_z))
    foreground_plane = bpy.context.active_object
    foreground_plane.name = "Foreground_Plane"

    direction_to_camera = Vector((camera_x - fg_x, camera_y - fg_y, camera_z - fg_z))
    direction_to_camera.normalize()
    foreground_plane.rotation_euler = direction_to_camera.to_track_quat('-Z', 'Y').to_euler()

    mat = bpy.data.materials.new(name="Foreground_Material")
    mat.use_nodes = True
    mat.blend_method = 'BLEND'
    foreground_plane.data.materials.append(mat)

    mat.node_tree.nodes.clear()
    bsdf = mat.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
    output = mat.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
    tex_image = mat.node_tree.nodes.new(type='ShaderNodeTexImage')
    tex_coord = mat.node_tree.nodes.new(type='ShaderNodeTexCoord')
    mapping = mat.node_tree.nodes.new(type='ShaderNodeMapping')

    bpy.ops.image.open(filepath=foreground_path)
    img = bpy.data.images[os.path.basename(foreground_path)]
    tex_image.image = img

    mat.node_tree.links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
    mat.node_tree.links.new(mapping.outputs['Vector'], tex_image.inputs['Vector'])
    mat.node_tree.links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])
    mat.node_tree.links.new(tex_image.outputs['Alpha'], bsdf.inputs['Alpha'])
    mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    # Random texture transformations
    scale_factor = random.uniform(0.8, 2.0)
    mapping.inputs['Scale'].default_value = (scale_factor, scale_factor, scale_factor)

    # Random rotation around Z-axis
    rotation_z = random.uniform(0, 2 * math.pi)
    mapping.inputs['Rotation'].default_value = (0, 0, rotation_z)

    # Random transparency level
    transparency = random.uniform(0.6, 0.9)
    bsdf.inputs['Alpha'].default_value = transparency

    return

# Randomize material properties for the model
def randomize_model_materials(obj):
    if not obj or not obj.data.materials:
        return

    # Generate random values for properties
    metallic = random.uniform(0.0, 0.25)
    roughness = random.uniform(0.1, 1.0)
    ior = random.uniform(1.0, 2.0)

    for material in obj.data.materials:
        if material and material.use_nodes:
            for node in material.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    node.inputs['Metallic'].default_value = metallic
                    node.inputs['Roughness'].default_value = roughness
                    node.inputs['IOR'].default_value = ior
                    material.blend_method = 'BLEND'
                    break

# Calculate bounding box of model
def get_model_bounding_box(obj, camera):
    scene = bpy.context.scene
    render = scene.render

    # Transform 3D points to 2D coordinates
    bbox_3d = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    screen_coords = []
    for point_3d in bbox_3d:
        co_2d = bpy_extras.object_utils.world_to_camera_view(scene, camera, point_3d)
        x = co_2d.x * render.resolution_x
        y = (1.0 - co_2d.y) * render.resolution_y
        if co_2d.z > 0:
            screen_coords.append((x, y))

    if not screen_coords:
        return None

    xs, ys = zip(*screen_coords)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(render.resolution_x, max_x)
    max_y = min(render.resolution_y, max_y)

    return min_x, min_y, max_x, max_y

# Create class mapping from folder names
def create_class_mapping(all_models):
    class_mapping = {}
    for i, (folder_name, _) in enumerate(all_models):
        class_mapping[folder_name] = i
    return class_mapping

# Initialize COCO file structure
def create_coco_structure(class_mapping):
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    for folder_name, class_id in class_mapping.items():
        coco_data["categories"].append({
            "id": class_id,
            "name": folder_name
        })

    return coco_data

# Generate COCO annotation
def generate_coco_annotation(class_id, image_id, annotation_id, bbox):
    min_x, min_y, max_x, max_y = bbox
    width = max_x - min_x
    height = max_y - min_y
    area = width * height

    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": class_id,
        "area": area,
        "bbox": [min_x, min_y, width, height],
        "iscrowd": 0
    }

    return annotation

# Save COCO data into a file
def save_coco_data(coco_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    annotations_file = os.path.join(output_dir, OUT_JSON)

    with open(annotations_file, 'w') as f:
        json.dump(coco_data, f, indent=2)

# Save the rendered scene as an image and generate COCO annotation
def save_scene(model_path, folder_name, i, obj, class_id, coco_data):
    global counter
    # Take a screenshot of the current scene
    bpy.context.scene.render.image_settings.file_format = 'JPEG'
    bpy.context.scene.render.resolution_x = IMAGE_RES[0]
    bpy.context.scene.render.resolution_y = IMAGE_RES[1]
    os.makedirs(folder_name, exist_ok=True)

    if len(model_paths) > 1:
        dae_filename = os.path.splitext(os.path.basename(model_path))[0]
        image_filename = f"img_{os.path.basename(folder_name)}_{dae_filename}_{i:05d}.jpg"
    else:
        image_filename = f"img_{os.path.basename(folder_name)}_{i:05d}.jpg"

    image_path = os.path.join(folder_name, image_filename)
    bpy.context.scene.render.filepath = image_path
    bpy.ops.render.render(write_still=True)
    camera = bpy.context.scene.camera

    # Add image info to COCO structure
    image_id = len(coco_data["images"]) + 1
    coco_data["images"].append({
        "id": image_id,
        "width": IMAGE_RES[0],
        "height": IMAGE_RES[1],
        "file_name": image_filename
    })

    # Add annotation to COCO structure
    bbox = get_model_bounding_box(obj, camera)
    coco_annotation = generate_coco_annotation(class_id, image_id, counter, bbox)
    coco_data["annotations"].append(coco_annotation)
    counter += 1

    return

all_models = find_models()
backgrounds = find_background()
foregrounds = find_foreground()

class_mapping = create_class_mapping(all_models)
coco_data = create_coco_structure(class_mapping)

# Create model images and annotations
for folder_name, model_paths in all_models:
    source_model_folder_name = folder_name
    folder_name = os.path.join(OUT_IMAGE, source_model_folder_name)
    class_id = class_mapping[source_model_folder_name]

    for model_path in model_paths:
        clear_scene()
        obj = import_model(model_path)

        model_size, model_center = get_model_info(obj)
        total_images_per_model = NUM_IMAGES // len(model_paths)
        total_loops = total_images_per_model // IMAGES_PER_LOOP

        for loop in range(total_loops):
            for image in range(IMAGES_PER_LOOP):
                randomize_model_materials(obj)
                camera_location = add_camera(model_size, model_center, image)
                add_light(model_size, camera_location)

                if backgrounds:
                    _, bg_texture = random.choice(backgrounds)
                    add_background(bg_texture)

                if foregrounds and random.random() < FOREGROUND_CHANCE:
                    _, fg_texture = random.choice(foregrounds)
                    add_foreground(model_size, model_center, camera_location, fg_texture)

                image_index = loop * IMAGES_PER_LOOP + image
                save_scene(model_path, folder_name, image_index, obj, class_id, coco_data)

# Generate background images for negative training
background_folder = os.path.join(OUT_IMAGE, "background")
os.makedirs(background_folder, exist_ok=True)

num_background_images = NUM_IMAGES // 10
for bg_img_index in range(num_background_images):
    clear_scene()

    # Add camera at random position
    camera_location = add_camera(50, Vector((0, 0, 0)), bg_img_index)
    add_light(50, camera_location)

    # Add background
    if backgrounds:
        _, bg_texture = random.choice(backgrounds)
        add_background(bg_texture)

    # Add foreground elements
    if foregrounds and random.random() < FOREGROUND_CHANCE:
        _, fg_texture = random.choice(foregrounds)
        add_foreground(50, Vector((0, 0, 0)), camera_location, fg_texture)

    bpy.context.scene.render.image_settings.file_format = 'JPEG'
    bpy.context.scene.render.resolution_x = IMAGE_RES[0]
    bpy.context.scene.render.resolution_y = IMAGE_RES[1]
    bg_image_filename = f"bg_{bg_img_index:05d}.jpg"
    bg_image_path = os.path.join(background_folder, bg_image_filename)
    bpy.context.scene.render.filepath = bg_image_path
    bpy.ops.render.render(write_still=True)

    # Add to COCO data
    image_id = len(coco_data["images"]) + 1
    coco_data["images"].append({
        "id": image_id,
        "width": IMAGE_RES[0],
        "height": IMAGE_RES[1],
        "file_name": bg_image_filename
    })

# Save COCO annotations into a file
save_coco_data(coco_data, BASE_DIR)
print("Dataset generation complete")
bpy.ops.wm.quit_blender()