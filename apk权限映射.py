# -*- coding: utf-8 -*-
"""
APK权限映射脚本
功能：
1. 读取APK提取的权限文件
2. 筛选系统权限
3. 使用 permission_knowledge.json 中的英文解释
4. 输出到 dataset/apk映射/应用名.txt

输出格式：
android.permission.CAMERA    Allows an application to access the camera device.
"""

import json
import re
from pathlib import Path


# =========================
# 路径配置
# =========================

BASE_DIR = Path("dataset")

APK_PERMISSION_DIR = BASE_DIR / "apk权限"
OUTPUT_DIR = BASE_DIR / "apk映射"

PERMISSION_KB_PATH = Path("dataset/permission_knowledge.json")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 提取 Description
# =========================

def extract_description(content: str) -> str:
    """
    从 permission_knowledge.json 的 content 中提取 Description
    """

    if not content:
        return ""

    match = re.search(r"Description:\s*(.+)", content)

    if match:
        desc = match.group(1).strip()

        # 去掉HTML标签
        desc = re.sub(r"<.*?>", "", desc)

        # 去掉多余空格
        desc = re.sub(r"\s+", " ", desc)

        return desc

    return ""


# =========================
# 加载权限知识库
# =========================

def load_permission_kb():
    """
    加载 permission_knowledge.json
    返回：
    {
        "android.permission.CAMERA": "Allows an application to access the camera device.",
        ...
    }
    """

    with open(PERMISSION_KB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    permission_map = {}

    for item in data:

        perm = item.get("permission", "").strip()

        content = item.get("content", "")

        desc = extract_description(content)

        if perm and desc:
            permission_map[perm] = desc

    return permission_map


# =========================
# 读取APK权限文件
# =========================

def load_apk_permissions(file_path: Path):
    """
    从APK权限文件读取权限列表
    """

    perms = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:

            line = line.strip()

            if not line:
                continue

            perms.append(line)

    return perms


# =========================
# 映射权限
# =========================

def map_permissions(permission_map):

    for apk_file in APK_PERMISSION_DIR.glob("*.txt"):

        app_name = apk_file.stem

        permissions = load_apk_permissions(apk_file)

        mapped_permissions = []

        for perm in permissions:

            if perm in permission_map:

                desc = permission_map[perm]

                mapped_permissions.append(f"{perm}\t{desc}")

        if not mapped_permissions:
            continue

        output_file = OUTPUT_DIR / f"{app_name}.txt"

        with open(output_file, "w", encoding="utf-8") as f:

            for line in mapped_permissions:
                f.write(line + "\n")

        print(f"✔ 已生成: {output_file}")


# =========================
# 主程序
# =========================

def main():

    print("加载权限知识库...")

    permission_map = load_permission_kb()

    print(f"权限知识条目数量: {len(permission_map)}")

    print("开始映射APK权限...")

    map_permissions(permission_map)

    print("完成")


if __name__ == "__main__":
    main()