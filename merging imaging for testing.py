import os
import shutil
import random

# paths
normal_dir = r"D:\myproject\oral-cancer-clinical-histopathological-image\Oral Cancer\oral_normal"
scc_dir = r"D:\myproject\oral-cancer-clinical-histopathological-image\Oral Cancer\oral_scc"
test_dir = r"D:\myproject\oral-cancer-clinical-histopathological-image\test_histopath"

os.makedirs(test_dir, exist_ok=True)

# list images
normal_imgs = [f for f in os.listdir(normal_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
scc_imgs = [f for f in os.listdir(scc_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# calculate 5%
normal_count = int(0.05 * len(normal_imgs))
scc_count = int(0.05 * len(scc_imgs))

# random sampling
test_normal = random.sample(normal_imgs, normal_count)
test_scc = random.sample(scc_imgs, scc_count)

# copy normal images
for img in test_normal:
    src = os.path.join(normal_dir, img)
    dst = os.path.join(test_dir, f"normal_{img}")
    shutil.copy(src, dst)

# copy scc images
for img in test_scc:
    src = os.path.join(scc_dir, img)
    dst = os.path.join(test_dir, f"scc_{img}")
    shutil.copy(src, dst)

print("✅ Test set created successfully")
print(f"Oral Normal images: {normal_count}")
print(f"Oral SCC images: {scc_count}")
print(f"Total Test images: {normal_count + scc_count}")
