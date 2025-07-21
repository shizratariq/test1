import os
import xml.etree.ElementTree as ET
import cv2

# === CONFIGURATION ===
xml_path = 'annotations.xml'
image_dir = 'images'
output_dir = 'yolo_labels'

# === Setup ===
print(f"📂 Parsing XML from: {xml_path}")
tree = ET.parse(xml_path)
root = tree.getroot()

image_filenames = set(os.listdir(image_dir))
print(f"🖼️ Total images found: {len(image_filenames)}")
print("🔍 Sample filenames:", list(image_filenames)[:5])

# === Make output directory ===
os.makedirs(output_dir, exist_ok=True)

def polyline_to_polygon(points, buffer_width=5):
    polygon = []
    for x, y in points:
        polygon.append((x - buffer_width, y))
    for x, y in reversed(points):
        polygon.append((x + buffer_width, y))
    return polygon

def normalize_points(points, img_w, img_h):
    return [(x / img_w, y / img_h) for x, y in points]

frame_labels = {}
tracks = root.findall('.//track')
print(f"🎯 Found {len(tracks)} tracks")

for track in tracks:
    label = track.get('label')
    if label is None or 'shaft' not in label.lower():
        continue

    for poly in track.findall('polyline'):
        frame = poly.get('frame')
        points_str = poly.get('points')
        outside = poly.get('outside')

        if outside == "1" or points_str is None:
            continue

        try:
            frame = int(frame)
            points = [(float(x), float(y)) for x, y in (p.split(',') for p in points_str.split(';'))]
        except Exception as e:
            print(f"❌ Frame {frame} point parse error: {e}")
            continue

        # Match file like frame_025480.jpg
        frame_str = f"{frame:06d}"  # zero-padded to 6 digits
        matched_file = None
        for fname in image_filenames:
            if f"frame_{frame_str}" in fname and fname.lower().endswith(('.jpg', '.png')):
                matched_file = fname
                break

        if not matched_file:
            print(f"⚠️ No image for frame {frame_str}")
            continue

        image_path = os.path.join(image_dir, matched_file)
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not load {matched_file}")
            continue

        image_height, image_width = img.shape[:2]

        polygon = polyline_to_polygon(points)
        polygon_normalized = normalize_points(polygon, image_width, image_height)
        flat_coords = [str(coord) for point in polygon_normalized for coord in point]
        yolo_line = '0 ' + ' '.join(flat_coords)

        if frame not in frame_labels:
            frame_labels[frame] = []
        frame_labels[frame].append(yolo_line)
        print(f"✅ Frame {frame} matched with {matched_file}")

# === Write YOLO label files ===
for frame, lines in frame_labels.items():
    frame_str = f"{frame:06d}"
    label_filename = f"frame_{frame_str}.txt"
    label_path = os.path.join(output_dir, label_filename)
    with open(label_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"💾 Saved: {label_path}")

print("\n✅ All done! YOLO labels are in:", output_dir)
