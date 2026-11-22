"""生成合成风景/自然图像作为训练集（无需外部下载）。

生成策略：
  - 渐变天空 + 地面（山林/草地）+ 简单几何树木和太阳
  - 多样化配色：日出、日落、晴天、雪景、沙漠、海洋等
  - 输出到 data/datasets/images/
"""
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image


def make_gradient_image(W, H):
    """生成纯渐变背景（用于小图测试）"""
    img = Image.new('RGB', (W, H))
    pixels = img.load()
    for y in range(H):
        for x in range(W):
            t = y / H
            r = int(50 + 150 * t)
            g = int(80 + 100 * (1 - abs(2*t - 1)))
            b = int(150 + 80 * (1 - t))
            pixels[x, y] = (r, g, b)
    return img


def paint_landscape(w, h):
    """绘制一幅简单的合成风景画。"""
    # 使用int32避免溢出，最后转uint8
    img = np.zeros((h, w, 3), dtype=np.int32)
    scene_type = random.choice(['mountain', 'sea', 'forest', 'desert', 'snow', 'sunset'])

    if scene_type == 'mountain':
        # 渐变天空
        for y in range(h // 3):
            t = y / (h / 3)
            img[y, :, 0] = int(30 + 80 * t)
            img[y, :, 1] = int(60 + 80 * t)
            img[y, :, 2] = int(160 + 60 * t)
        # 山脉
        cx = w // 2
        for y in range(h // 3, h // 2):
            offset = int((y - h // 3) * 0.5)
            for dy in range(-40, 40):
                mx = int(cx + dy + random.randint(-3, 3))
                if 0 <= mx < w:
                    shade = 80 + int(random.gauss(0, 15))
                    img[y, mx, :] = [shade, shade + 10, shade + 5]
        # 雪地山顶
        for y in range(h // 3, h // 3 + 30):
            offset = int((y - h // 3) * 0.3)
            for dx in range(max(0, cx - 60 + offset), min(w, cx + 60 - offset)):
                img[y, dx, :] = [230, 235, 250]
        # 草地
        for y in range(h // 2, h):
            t = (y - h // 2) / (h // 2)
            img[y, :, 0] = int(40 + 30 * t)
            img[y, :, 1] = int(90 + 40 * t)
            img[y, :, 2] = int(30 + 20 * t)
        # 几棵小树
        for _ in range(random.randint(3, 8)):
            tx = random.randint(20, w - 20)
            ty = random.randint(h // 2 + 10, h - 30)
            th = random.randint(15, 35)
            for dy in range(th):
                width = max(1, int((th - dy) * 0.6))
                for dx in range(-width, width):
                    px, py = tx + dx, ty - dy
                    if 0 <= px < w and 0 <= py < h:
                        img[py, px, 1] = min(255, img[py, px, 1] + random.randint(20, 50))
                        img[py, px, 0] = max(0, img[py, px, 0] - random.randint(10, 30))

    elif scene_type == 'sea':
        for y in range(h // 3):
            t = y / (h / 3)
            img[y, :, 0] = int(200 - 80 * t)
            img[y, :, 1] = int(200 - 60 * t)
            img[y, :, 2] = int(255 - 40 * t)
        sun_y = h // 4
        for sx in range(w):
            dist = abs(sx - w // 2) / (w // 2)
            brightness = int(255 * (1 - dist * 0.7))
            img[sun_y, sx, :] = [brightness, brightness - 30, brightness - 60]
        for y in range(h // 3, h):
            t = (y - h // 3) / (2 * h // 3)
            shimmer = int(random.gauss(0, 5))
            img[y, :, 0] = int(20 + 40 * t + shimmer)
            img[y, :, 1] = int(50 + 80 * t + shimmer)
            img[y, :, 2] = int(120 + 60 * t + shimmer)
        # 沙滩
        sand_y = h // 3 - 15
        img[sand_y:sand_y + 15, :, :] = [int(210 + random.gauss(0, 8)),
                                          int(190 + random.gauss(0, 8)),
                                          int(150 + random.gauss(0, 8))]

    elif scene_type == 'forest':
        for y in range(h // 2):
            t = y / (h // 2)
            img[y, :, 0] = int(30 + 50 * t)
            img[y, :, 1] = int(60 + 40 * t)
            img[y, :, 2] = int(40 + 30 * t)
        for y in range(h // 2, h):
            t = (y - h // 2) / (h // 2)
            img[y, :, 0] = int(20 + 20 * t)
            img[y, :, 1] = int(70 + 30 * t)
            img[y, :, 2] = int(20 + 15 * t)
        for _ in range(random.randint(8, 20)):
            tx = random.randint(10, w - 10)
            tree_h = random.randint(40, 80)
            ty = random.randint(h // 2 + 5, h - 5)
            trunk_w = random.randint(2, 5)
            crown_r = random.randint(8, 20)
            for dy in range(tree_h):
                layer_w = max(trunk_w, int(crown_r * (1 - dy / tree_h)))
                for dx in range(-layer_w, layer_w):
                    px, py = tx + dx, ty - dy
                    if 0 <= px < w and 0 <= py < h:
                        if dy > tree_h * 0.3:
                            img[py, px, 1] = min(255, img[py, px, 1] + random.randint(30, 60))
                            img[py, px, 0] = max(0, img[py, px, 0] - random.randint(10, 25))
                        else:
                            img[py, px, :] = [int(80 + random.gauss(0, 15)),
                                              int(50 + random.gauss(0, 10)),
                                              int(30 + random.gauss(0, 8))]

    elif scene_type == 'desert':
        for y in range(h // 2):
            t = y / (h // 2)
            img[y, :, 0] = int(220 - 60 * t)
            img[y, :, 1] = int(180 - 40 * t)
            img[y, :, 2] = int(120 - 30 * t)
        for y in range(h // 2, h):
            t = (y - h // 2) / (h // 2)
            img[y, :, 0] = int(200 + 30 * t)
            img[y, :, 1] = int(170 + 20 * t)
            img[y, :, 2] = int(110 + 20 * t)
        # 沙丘起伏
        for x in range(w):
            ridge = int(np.sin(x * 0.02) * 20 + np.sin(x * 0.05) * 10)
            ry = h // 2 + ridge
            for dy in range(-5, 5):
                py = int(ry + dy)
                if 0 <= py < h:
                    shade = int(20 + abs(dy) * 5)
                    img[py, x, :] = [min(255, img[py, x, 0] + shade),
                                     min(255, img[py, x, 1] + shade - 5),
                                     min(255, img[py, x, 2] + shade - 10)]

    elif scene_type == 'snow':
        for y in range(h // 3):
            t = y / (h // 3)
            img[y, :, 0] = int(180 + 50 * t)
            img[y, :, 1] = int(200 + 30 * t)
            img[y, :, 2] = int(230 + 15 * t)
        for y in range(h // 3, h):
            img[y, :, :] = [int(220 + random.gauss(0, 5)),
                            int(225 + random.gauss(0, 5)),
                            int(235 + random.gauss(0, 5))]
        # 雪覆盖的松树
        for _ in range(random.randint(5, 15)):
            tx = random.randint(20, w - 20)
            ty = random.randint(h // 3 + 20, h - 20)
            th = random.randint(30, 60)
            for dy in range(th):
                layer_r = max(2, int((th - dy) * 0.4))
                snow_line = max(0, dy - 8)
                for dx in range(-layer_r, layer_r):
                    px, py = tx + dx, ty - dy
                    if 0 <= px < w and 0 <= py < h:
                        if dy < snow_line:
                            img[py, px, :] = [245, 248, 255]
                        else:
                            img[py, px, 1] = min(255, img[py, px, 1] + random.randint(20, 40))

    elif scene_type == 'sunset':
        colors = [(255, 140, 50), (255, 80, 60), (200, 50, 80),
                  (120, 40, 100), (60, 30, 80)]
        band_h = h // len(colors)
        for i, c in enumerate(colors):
            y_start = i * band_h
            y_end = y_start + band_h
            for y in range(y_start, min(y_end, h)):
                t = (y - y_start) / band_h
                j = min(i + 1, len(colors) - 1)
                next_c = colors[j]
                for ch in range(3):
                    val = int(c[ch] * (1 - t) + next_c[ch] * t)
                    img[y, :, ch] = val
        # 太阳
        sy = h // 3
        for r in range(60, 0, -1):
            intensity = r / 60
            for dy in range(-r, r):
                for dx in range(-r, r):
                    if dx*dx + dy*dy <= r*r:
                        px, py = w // 2 + dx, sy + dy
                        if 0 <= px < w and 0 <= py < h:
                            for ch in range(3):
                                img[py, px, ch] = int(min(255, img[py, px, ch] + 100 * intensity))
        # 水面倒影
        for y in range(h // 2, h):
            t = (y - h // 2) / (h // 2)
            shade_factor = 1 - t * 0.5
            img[y, :, 0] = (img[y, :, 0] * shade_factor).astype(np.uint8)
            img[y, :, 1] = (img[y, :, 1] * shade_factor).astype(np.uint8)
            img[y, :, 2] = (img[y, :, 2] * shade_factor).astype(np.uint8)
        # 波浪线
        for y in range(h // 2, h, 3):
            for x in range(0, w, 15):
                wave = int(np.sin(x * 0.05 + y * 0.1) * 3)
                if 0 <= x + wave < w:
                    img[y, x + wave, 1] = min(255, img[y, x + wave, 1] + 20)

    # 加轻微噪点让图片更真实
    noise = np.random.randint(-8, 8, size=img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(img)


def main():
    # Use temp path to avoid Chinese path encoding issues
    out_dir = Path("C:/temp-prism-data/datasets/images")
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = [(256, 256), (384, 256), (256, 384), (512, 256), (256, 512)]
    count = 200
    print(f"生成 {count} 张合成风景图到 {out_dir}")

    for i in range(count):
        w, h = random.choice(sizes)
        img = paint_landscape(w, h)
        path = out_dir / f"image_{i:04d}.jpg"
        img.save(path, quality=95)
        if (i + 1) % 50 == 0:
            print(f"  已生成 {i + 1}/{count}")

    print(f"完成！共 {len(list(out_dir.glob('*.jpg')))} 张图片")


if __name__ == "__main__":
    main()
