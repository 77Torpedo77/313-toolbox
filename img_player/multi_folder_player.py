#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Folder Image Player
用于同步播放多个文件夹中的图片序列，并提供轨迹评估可视化功能。

主要功能:
- YAML 配置文件加载: 支持图片文件夹、多条轨迹及真值 (GT) 文件的一键配置。
- 多路同步显示: 所有窗口根据帧索引同步播放。
- 轨迹评估对比: 实时显示每条轨迹相对于前一帧的运动 (ΔT 位移 & ΔR 旋转)，并与 GT 对比。
- 动态重载支持: 点击 "🔄 重载" 按钮即可刷新 config.yaml 及本地数据，无需重启软件。

使用说明:
1. 运行方式: 直接运行 `python multi_folder_player.py`。默认加载同目录 `config.yaml`。
2. 数据配置 (config.yaml):
   - folders: 图片目录列表。
   - trajectories: TUM 格式轨迹文件列表（第一条将用于运动数据展示）。
   - ground_truth: 真值轨迹路径（白色线条显示）。
   - ground_truth_frame_offset: 对齐补丁。例如 Est 帧 0 对应 GT 帧 1 时，设为 1。
3. 界面控制:
   - 🔄 重载: 刷新数据和配置（自动清除缓存）。
   - 空格/按键: 播放/暂停控制。
   - 滑动条: 快速跳转帧索引。
   - 轨迹视图: 滚轮缩放，左键拖动观察 X-Z 平面轨迹。
"""

import os
import sys
import re
import argparse
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import math
from typing import List, Optional, Callable, Dict, Tuple
import colorsys
from collections import OrderedDict
import yaml
import numpy as np

# 默认配置文件路径，可直接运行脚本
DEFAULT_CONFIG_PATH = r"D:\tools\313-toolbox\img_player\config.yaml"


class LRUCache:
    """LRU缓存，用于缓存PIL Image对象"""
    
    def __init__(self, max_size: int = 50):
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            self.cache[key] = value
    
    def clear(self):
        self.cache.clear()


class Trajectory:
    """轨迹数据类 - 支持位姿和相对运动计算"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.name = os.path.basename(file_path)
        self.frames: List[int] = []
        self.positions: List[Tuple[float, float, float]] = []
        self.quaternions: List[Tuple[float, float, float, float]] = []  # (w, x, y, z)
        self.frame_to_idx: Dict[int, int] = {}
        self.load_trajectory()
    
    def load_trajectory(self):
        if not os.path.isfile(self.file_path):
            return
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 4:
                        frame_idx = int(parts[0])
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        
                        # 加载四元数 (w, x, y, z)
                        if len(parts) >= 8:
                            qw, qx, qy, qz = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                        else:
                            qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0  # 默认单位四元数
                        
                        self.frame_to_idx[frame_idx] = len(self.frames)
                        self.frames.append(frame_idx)
                        self.positions.append((x, y, z))
                        self.quaternions.append((qw, qx, qy, qz))
        except Exception as e:
            print(f"加载轨迹文件失败 {self.file_path}: {e}")
    
    def get_position_by_frame(self, frame_idx: int) -> Optional[int]:
        return self.frame_to_idx.get(frame_idx)
    
    def get_relative_motion(self, frame_idx: int) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """
        计算相对于上一帧的运动
        返回: (translation, axis, angle_deg) 或 None
        """
        pos_idx = self.frame_to_idx.get(frame_idx)
        if pos_idx is None or pos_idx == 0:
            return None
        
        # 获取当前帧和前一帧的实际帧号
        curr_frame = self.frames[pos_idx]
        prev_frame = self.frames[pos_idx - 1]
        
        # 确保是连续帧 (帧号差为1)，否则相对运动无意义
        if curr_frame - prev_frame != 1:
            return None
        
        # 当前帧和上一帧的位姿
        t_curr = np.array(self.positions[pos_idx])
        t_prev = np.array(self.positions[pos_idx - 1])
        q_curr = np.array(self.quaternions[pos_idx])
        q_prev = np.array(self.quaternions[pos_idx - 1])
        
        # 计算相对平移 (在世界坐标系下)
        delta_t = t_curr - t_prev
        
        # 计算相对旋转: q_rel = q_curr * q_prev^{-1}
        q_prev_inv = quaternion_inverse(q_prev)
        q_rel = quaternion_multiply(q_curr, q_prev_inv)
        
        # 四元数转轴角
        axis, angle_deg = quaternion_to_axis_angle(q_rel)
        
        return (delta_t, axis, angle_deg)
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        if not self.positions:
            return (0, 1, 0, 1)
        
        x_vals = [p[0] for p in self.positions]
        z_vals = [p[2] for p in self.positions]
        
        x_min, x_max = min(x_vals), max(x_vals)
        z_min, z_max = min(z_vals), max(z_vals)
        
        if x_max - x_min < 1:
            x_min -= 0.5
            x_max += 0.5
        if z_max - z_min < 1:
            z_min -= 0.5
            z_max += 0.5
        
        return (x_min, x_max, z_min, z_max)


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """四元数求逆 (w, x, y, z)"""
    w, x, y, z = q
    norm_sq = w*w + x*x + y*y + z*z
    if norm_sq < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.array([w, -x, -y, -z]) / norm_sq


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """四元数乘法 (w, x, y, z)"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quaternion_to_axis_angle(q: np.ndarray) -> Tuple[np.ndarray, float]:
    """四元数转轴角表示 (w, x, y, z) -> (axis, angle_deg)"""
    w, x, y, z = q
    # 归一化
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm < 1e-10:
        return np.array([0.0, 0.0, 1.0]), 0.0
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # 确保 w >= 0 以取最短路径
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    
    # 计算角度
    angle_rad = 2.0 * math.acos(min(1.0, max(-1.0, w)))
    angle_deg = math.degrees(angle_rad)
    
    # 计算轴
    sin_half = math.sin(angle_rad / 2.0)
    if abs(sin_half) < 1e-10:
        return np.array([0.0, 0.0, 1.0]), 0.0
    
    axis = np.array([x, y, z]) / sin_half
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        return np.array([0.0, 0.0, 1.0]), 0.0
    axis = axis / axis_norm
    
    return axis, angle_deg


class TrajectoryPanel(ttk.Frame):
    """轨迹可视化面板 - 支持轨迹与真值相对运动对比"""
    
    def __init__(self, parent, trajectories: List[Trajectory], panel_size: tuple = (400, 300),
                 on_close: Callable = None, ground_truth: Optional[Trajectory] = None,
                 gt_frame_offset: int = 0):
        super().__init__(parent, relief="groove", borderwidth=2)
        
        self.trajectories = trajectories
        self.ground_truth = ground_truth  # 真值轨迹用于对比
        self.gt_frame_offset = gt_frame_offset  # 真值帧偏移量
        self.panel_width, self.panel_height = panel_size
        self.current_frame_index = 0
        self.on_close = on_close
        self.current_image: Optional[ImageTk.PhotoImage] = None
        self.base_image: Optional[Image.Image] = None
        self.last_size = (0, 0)
        self.screen_points = []
        
        self.colors = self._generate_colors(len(trajectories))
        
        self.title_frame = ttk.Frame(self)
        self.title_frame.pack(fill='x', padx=2, pady=2)
        
        self.title_label = ttk.Label(
            self.title_frame, text="轨迹视图 (X-Z平面)", 
            font=('Arial', 9, 'bold'), anchor='w'
        )
        self.title_label.pack(side='left', fill='x', expand=True)
        
        self.close_btn = ttk.Button(self.title_frame, text="✕", width=3, command=self._close_panel)
        self.close_btn.pack(side='right', padx=1)
        
        self.canvas = tk.Canvas(self, width=self.panel_width, height=self.panel_height,
                                bg='#1a1a2e', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True, padx=2, pady=2)
        
        self.legend_frame = ttk.Frame(self)
        self.legend_frame.pack(fill='x', padx=2, pady=2)
        
        self._create_legend()
        self._precompute_trajectory_data()
    
    def _generate_colors(self, n: int) -> List[str]:
        colors = []
        for i in range(n):
            hue = i / max(n, 1)
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            colors.append('#{:02x}{:02x}{:02x}'.format(
                int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))
        return colors if colors else ['#00ff00']
    
    def _create_legend(self):
        for widget in self.legend_frame.winfo_children():
            widget.destroy()
        
        for i, traj in enumerate(self.trajectories):
            color = self.colors[i] if i < len(self.colors) else '#ffffff'
            color_frame = tk.Frame(self.legend_frame, bg=color, width=12, height=12)
            color_frame.pack(side='left', padx=2)
            color_frame.pack_propagate(False)
            
            name = traj.name[:20] + '...' if len(traj.name) > 20 else traj.name
            ttk.Label(self.legend_frame, text=name, font=('Arial', 7)).pack(side='left', padx=(0, 8))
        
        # GT 图例
        if self.ground_truth:
            color_frame = tk.Frame(self.legend_frame, bg='#00ff00', width=12, height=12)
            color_frame.pack(side='left', padx=2)
            color_frame.pack_propagate(False)
            ttk.Label(self.legend_frame, text="GT", font=('Arial', 7, 'bold')).pack(side='left', padx=(0, 8))
    
    def _precompute_trajectory_data(self):
        all_bounds = [t.get_bounds() for t in self.trajectories if t.positions]
        # 包含 GT 轨迹的边界
        if self.ground_truth and self.ground_truth.positions:
            all_bounds.append(self.ground_truth.get_bounds())
        
        if not all_bounds:
            self.global_bounds = (0, 1, 0, 1)
        else:
            self.global_bounds = (
                min(b[0] for b in all_bounds), max(b[1] for b in all_bounds),
                min(b[2] for b in all_bounds), max(b[3] for b in all_bounds)
            )
    
    def _close_panel(self):
        if self.on_close:
            self.on_close(self)
    
    def display_frame(self, frame_index: int):
        self.current_frame_index = frame_index
        self._draw_trajectories()
    
    def _create_base_image(self, canvas_width: int, canvas_height: int):
        img = Image.new('RGB', (canvas_width, canvas_height), '#1a1a2e')
        draw = ImageDraw.Draw(img)
        
        x_min, x_max, z_min, z_max = self.global_bounds
        margin = 30
        plot_width = canvas_width - 2 * margin
        plot_height = canvas_height - 2 * margin
        
        x_range, z_range = x_max - x_min, z_max - z_min
        scale = min(plot_width / x_range if x_range > 0 else 1,
                   plot_height / z_range if z_range > 0 else 1)
        
        offset_x = margin + (plot_width - x_range * scale) / 2
        offset_z = margin + (plot_height - z_range * scale) / 2
        
        def world_to_screen(x, z):
            return (offset_x + (x - x_min) * scale, offset_z + (z_max - z) * scale)
        
        grid_color = '#2d2d44'
        for i in range(0, canvas_width, 50):
            draw.line([(i, 0), (i, canvas_height)], fill=grid_color, width=1)
        for i in range(0, canvas_height, 50):
            draw.line([(0, i), (canvas_width, i)], fill=grid_color, width=1)
        
        self.screen_points = []
        
        for traj_idx, traj in enumerate(self.trajectories):
            if not traj.positions:
                self.screen_points.append([])
                continue
            
            color = self.colors[traj_idx] if traj_idx < len(self.colors) else '#ffffff'
            points = [world_to_screen(pos[0], pos[2]) for pos in traj.positions]
            self.screen_points.append(points)
            
            if len(points) >= 2:
                draw.line(points, fill=color, width=2)
            
            if points:
                start = points[0]
                draw.ellipse([start[0]-5, start[1]-5, start[0]+5, start[1]+5], 
                            fill='#00ff00', outline='white')
        
        # 绘制 GT 轨迹 (亮绿色，更粗的线)
        self.screen_points_gt = []
        if self.ground_truth and self.ground_truth.positions:
            gt_points = [world_to_screen(pos[0], pos[2]) for pos in self.ground_truth.positions]
            self.screen_points_gt = gt_points
            
            if len(gt_points) >= 2:
                draw.line(gt_points, fill='#00ff00', width=3)  # 亮绿色，更粗
            
            if gt_points:
                start = gt_points[0]
                draw.ellipse([start[0]-6, start[1]-6, start[0]+6, start[1]+6], 
                            fill='#00ff00', outline='white')
        
        draw.text((5, canvas_height - 15), "X", fill='#888888')
        draw.text((canvas_width - 15, 5), "Z", fill='#888888')
        
        return img
    
    def _draw_trajectories(self):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width < 10 or canvas_height < 10:
            return
        
        current_size = (canvas_width, canvas_height)
        if self.base_image is None or self.last_size != current_size:
            self.base_image = self._create_base_image(canvas_width, canvas_height)
            self.last_size = current_size
        
        img = self.base_image.copy()
        draw = ImageDraw.Draw(img)
        
        for traj_idx, traj in enumerate(self.trajectories):
            if not traj.positions or traj_idx >= len(self.screen_points):
                continue
            
            points = self.screen_points[traj_idx]
            pos_idx = traj.get_position_by_frame(self.current_frame_index)
            
            if pos_idx is not None and pos_idx < len(points):
                curr = points[pos_idx]
                draw.ellipse([curr[0]-8, curr[1]-8, curr[0]+8, curr[1]+8], 
                            fill='#ff3366', outline='white', width=2)
                
                if pos_idx < len(points) - 1:
                    next_pt = points[pos_idx + 1]
                    dx, dz = next_pt[0] - curr[0], next_pt[1] - curr[1]
                    length = math.sqrt(dx*dx + dz*dz)
                    if length > 0:
                        dx, dz = dx/length * 15, dz/length * 15
                        draw.line([curr, (curr[0]+dx, curr[1]+dz)], fill='white', width=2)
        
        # 绘制帧信息
        draw.text((5, 5), f"Frame: {self.current_frame_index}", fill='white')
        
        # 绘制相对运动信息
        y_offset = 25
        
        # 所有轨迹的估计值 (颜色与图例对应)
        for traj_idx, traj in enumerate(self.trajectories):
            motion = traj.get_relative_motion(self.current_frame_index)
            if motion:
                dt, axis, angle = motion
                color = self.colors[traj_idx] if traj_idx < len(self.colors) else '#ffffff'
                # 使用短名称标识
                short_name = f"Est{traj_idx}"
                draw.text((5, y_offset), f"{short_name} ΔT: [{dt[0]:.3f}, {dt[1]:.3f}, {dt[2]:.3f}]", fill=color)
                y_offset += 15
                draw.text((5, y_offset), f"{short_name} ΔR: θ={angle:.2f}° axis=[{axis[0]:.2f},{axis[1]:.2f},{axis[2]:.2f}]", fill=color)
                y_offset += 18
        
        # 真值 (应用帧偏移量对齐)
        if self.ground_truth:
            gt_frame = self.current_frame_index + self.gt_frame_offset
            gt_motion = self.ground_truth.get_relative_motion(gt_frame)
            if gt_motion:
                dt, axis, angle = gt_motion
                draw.text((5, y_offset), f"GT  ΔT: [{dt[0]:.3f}, {dt[1]:.3f}, {dt[2]:.3f}]", fill='#00ff88')
                y_offset += 15
                draw.text((5, y_offset), f"GT  ΔR: θ={angle:.2f}° axis=[{axis[0]:.2f},{axis[1]:.2f},{axis[2]:.2f}]", fill='#00ff88')
        
        self.current_image = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.current_image, anchor='nw')
    
    def update_panel_size(self, width: int, height: int):
        self.panel_width, self.panel_height = width, height
        self.canvas.config(width=width, height=height)
        self.base_image = None


class ImageFolder:
    """表示一个图片文件夹"""
    
    SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')
    
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.folder_name = os.path.basename(folder_path)
        self.image_files: List[str] = []
        self.load_images()
    
    def load_images(self):
        if not os.path.isdir(self.folder_path):
            return
        
        all_files = os.listdir(self.folder_path)
        image_files = [f for f in all_files if f.lower().endswith(self.SUPPORTED_EXTENSIONS)]
        image_files.sort(key=lambda x: self._natural_sort_key(x))
        self.image_files = [os.path.join(self.folder_path, f) for f in image_files]
    
    def _natural_sort_key(self, s: str):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
    
    def get_image_count(self) -> int:
        return len(self.image_files)
    
    def get_image_path(self, index: int) -> Optional[str]:
        if 0 <= index < len(self.image_files):
            return self.image_files[index]
        return None
    
    def get_image_filename(self, index: int) -> Optional[str]:
        path = self.get_image_path(index)
        return os.path.basename(path) if path else None


class ImagePanel(ttk.Frame):
    """单个图片显示面板"""
    
    pil_cache = LRUCache(max_size=100)
    
    def __init__(self, parent, folder: ImageFolder, panel_size: tuple = (320, 240),
                 on_close: Callable = None):
        super().__init__(parent, relief="groove", borderwidth=2)
        
        self.folder = folder
        self.panel_width, self.panel_height = panel_size
        self.current_image: Optional[ImageTk.PhotoImage] = None
        self.current_frame_index = 0
        self.on_close = on_close
        
        self.title_frame = ttk.Frame(self)
        self.title_frame.pack(fill='x', padx=2, pady=2)
        
        self.title_label = ttk.Label(self.title_frame, text=folder.folder_name, 
                                     font=('Arial', 9, 'bold'), anchor='w')
        self.title_label.pack(side='left', fill='x', expand=True)
        
        self.close_btn = ttk.Button(self.title_frame, text="✕", width=3, command=self._close_panel)
        self.close_btn.pack(side='right', padx=1)
        
        self.canvas = tk.Canvas(self, width=self.panel_width, height=self.panel_height,
                                bg='black', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True, padx=2, pady=2)
        
        self.info_frame = ttk.Frame(self)
        self.info_frame.pack(fill='x', padx=2, pady=2)
        
        self.frame_info_label = ttk.Label(self.info_frame, text=f"0/{folder.get_image_count()}", 
                                          anchor='w', width=10)
        self.frame_info_label.pack(side='left')
        
        self.filename_label = ttk.Label(self.info_frame, text="", anchor='e', font=('Consolas', 8))
        self.filename_label.pack(side='right', fill='x', expand=True)
    
    def _close_panel(self):
        if self.on_close:
            self.on_close(self)
    
    def display_frame(self, frame_index: int):
        """显示指定索引的图片"""
        self.current_frame_index = frame_index
        image_path = self.folder.get_image_path(frame_index)
        
        if image_path is None:
            self.canvas.delete("all")
            self.canvas.create_text(self.panel_width // 2, self.panel_height // 2,
                                   text="No Image", fill='gray', font=('Arial', 12))
            self.filename_label.config(text="")
            return
        
        try:
            cache_key = f"{image_path}_{self.panel_width}_{self.panel_height}"
            cached_pil = ImagePanel.pil_cache.get(cache_key)
            
            if cached_pil is not None:
                pil_img = cached_pil
            else:
                pil_img = Image.open(image_path)
                pil_img = self._resize_to_fit(pil_img)
                ImagePanel.pil_cache.put(cache_key, pil_img)
            
            self.current_image = ImageTk.PhotoImage(pil_img)
            
            self.canvas.delete("all")
            self.canvas.create_image(self.panel_width // 2, self.panel_height // 2,
                                    image=self.current_image, anchor='center')
            
            self.frame_info_label.config(text=f"{frame_index + 1}/{self.folder.get_image_count()}")
            self.filename_label.config(text=self.folder.get_image_filename(frame_index) or "")
                
        except Exception as e:
            self.canvas.delete("all")
            self.canvas.create_text(self.panel_width // 2, self.panel_height // 2,
                                   text=f"Error: {str(e)[:20]}", fill='red', font=('Arial', 10))
    
    def _resize_to_fit(self, img: Image.Image) -> Image.Image:
        img_width, img_height = img.size
        ratio = min(self.panel_width / img_width, self.panel_height / img_height)
        return img.resize((int(img_width * ratio), int(img_height * ratio)), Image.Resampling.BILINEAR)
    
    def update_panel_size(self, width: int, height: int):
        self.panel_width, self.panel_height = width, height
        self.canvas.config(width=width, height=height)


class MultiFolderPlayer(ttk.Frame):
    """多文件夹图片播放器主界面"""
    
    def __init__(self, parent, folders: List[str] = None, trajectories: List[str] = None,
                 ground_truth: str = None, gt_frame_offset: int = 0, config_path: str = None):
        super().__init__(parent)
        
        self.parent = parent
        self.folders: List[ImageFolder] = []
        self.panels: List[ImagePanel] = []
        self.trajectories: List[Trajectory] = []
        self.ground_truth: Optional[Trajectory] = None
        self.gt_frame_offset = gt_frame_offset  # 真值帧偏移量
        self.trajectory_panel: Optional[TrajectoryPanel] = None
        self.current_frame = 0
        self.max_frames = 0
        self.is_playing = False
        self.play_interval = 100
        self.play_job = None
        
        # 存储配置路径用于重载
        self.config_path = config_path
        self.initial_folders = folders
        self.initial_trajectories = trajectories
        self.initial_ground_truth = ground_truth
        
        self.cols, self.rows = 1, 1
        self.panel_width, self.panel_height = 320, 240
        
        self._create_ui()
        
        if folders or trajectories or ground_truth:
            self.load_data(folders or [], trajectories or [], ground_truth)
    
    def _create_ui(self):
        self.pack(fill='both', expand=True)
        
        self.control_frame = ttk.Frame(self)
        self.control_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(self.control_frame, text="添加文件夹", command=self._add_folder).pack(side='left', padx=2)
        ttk.Button(self.control_frame, text="🔄 重载", command=self._reload_data).pack(side='left', padx=2)
        ttk.Button(self.control_frame, text="清空", command=self._clear_all).pack(side='left', padx=2)
        
        ttk.Separator(self.control_frame, orient='vertical').pack(side='left', fill='y', padx=10)
        
        ttk.Button(self.control_frame, text="◀◀", width=4, command=self._prev_frame).pack(side='left', padx=2)
        self.play_btn = ttk.Button(self.control_frame, text="▶", width=4, command=self._toggle_play)
        self.play_btn.pack(side='left', padx=2)
        ttk.Button(self.control_frame, text="▶▶", width=4, command=self._next_frame).pack(side='left', padx=2)
        
        self.frame_var = tk.IntVar(value=0)
        self.frame_slider = ttk.Scale(self.control_frame, from_=0, to=100, orient='horizontal',
                                      variable=self.frame_var, command=self._on_slider_change)
        self.frame_slider.pack(side='left', fill='x', expand=True, padx=10)
        
        self.frame_label = ttk.Label(self.control_frame, text="0/0", width=12)
        self.frame_label.pack(side='left', padx=5)
        
        ttk.Label(self.control_frame, text="速度:").pack(side='left', padx=2)
        self.speed_var = tk.StringVar(value="10")
        self.speed_combo = ttk.Combobox(self.control_frame, textvariable=self.speed_var,
                                        values=["1", "2", "5", "10", "15", "20", "30"], width=4, state='readonly')
        self.speed_combo.pack(side='left', padx=2)
        self.speed_combo.bind('<<ComboboxSelected>>', self._on_speed_change)
        ttk.Label(self.control_frame, text="FPS").pack(side='left')
        
        self.canvas_container = ttk.Frame(self)
        self.canvas_container.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.v_scrollbar = ttk.Scrollbar(self.canvas_container, orient='vertical')
        self.h_scrollbar = ttk.Scrollbar(self.canvas_container, orient='horizontal')
        self.main_canvas = tk.Canvas(self.canvas_container, xscrollcommand=self.h_scrollbar.set,
                                     yscrollcommand=self.v_scrollbar.set, bg='#2b2b2b')
        
        self.h_scrollbar.config(command=self.main_canvas.xview)
        self.v_scrollbar.config(command=self.main_canvas.yview)
        
        self.v_scrollbar.pack(side='right', fill='y')
        self.h_scrollbar.pack(side='bottom', fill='x')
        self.main_canvas.pack(side='left', fill='both', expand=True)
        
        self.panel_container = ttk.Frame(self.main_canvas)
        self.main_canvas.create_window((0, 0), window=self.panel_container, anchor='nw')
        
        self.panel_container.bind('<Configure>', lambda e: self.main_canvas.configure(
            scrollregion=self.main_canvas.bbox("all")))
        self.main_canvas.bind('<Configure>', lambda e: self._recalculate_layout() if self.folders or self.trajectories else None)
        
        self.parent.bind('<Left>', lambda e: self._prev_frame())
        self.parent.bind('<Right>', lambda e: self._next_frame())
        self.parent.bind('<space>', lambda e: self._toggle_play())
        self.parent.bind('<Home>', lambda e: self._goto_first_frame())
        self.parent.bind('<End>', lambda e: self._goto_last_frame())
        
        self.status_frame = ttk.Frame(self)
        self.status_frame.pack(fill='x', padx=5, pady=2)
        self.status_label = ttk.Label(self.status_frame, text="就绪 | ←/→ 前后帧, 空格 播放/暂停")
        self.status_label.pack(side='left')
    
    def _recalculate_layout(self):
        total = len(self.panels) + (1 if self.trajectory_panel else 0)
        if total == 0:
            return
        
        w, h = self.main_canvas.winfo_width(), self.main_canvas.winfo_height()
        if w < 100 or h < 100:
            return
        
        best_cols, best_area = 1, 0
        for cols in range(1, total + 1):
            rows = math.ceil(total / cols)
            pw, ph = (w - 20) // cols, (h - 20) // rows
            if pw > ph * 2: pw = ph * 2
            if ph > pw * 2: ph = pw * 2
            area = pw * ph
            if area > best_area and pw >= 200 and ph >= 150:
                best_area, best_cols = area, cols
        
        self.cols = best_cols
        self.rows = math.ceil(total / self.cols)
        self.panel_width = max(200, (w - 20) // self.cols - 10)
        self.panel_height = max(150, self.panel_width * 3 // 4)
        
        ImagePanel.pil_cache.clear()
        
        for panel in self.panels:
            panel.update_panel_size(self.panel_width, self.panel_height)
        if self.trajectory_panel:
            self.trajectory_panel.update_panel_size(self.panel_width, self.panel_height)
        
        self._display_current_frame()
    
    def load_data(self, folder_paths: List[str], trajectory_paths: List[str], 
                  ground_truth_path: str = None):
        self._clear_all()
        
        for path in folder_paths:
            if os.path.isdir(path):
                folder = ImageFolder(path)
                if folder.get_image_count() > 0:
                    self.folders.append(folder)
                else:
                    print(f"警告: 文件夹 {path} 中没有图片")
        
        for path in trajectory_paths:
            if os.path.isfile(path):
                traj = Trajectory(path)
                if traj.positions:
                    self.trajectories.append(traj)
        
        # 加载真值轨迹
        if ground_truth_path and os.path.isfile(ground_truth_path):
            self.ground_truth = Trajectory(ground_truth_path)
            if not self.ground_truth.positions:
                self.ground_truth = None
                print(f"警告: 真值文件无效 {ground_truth_path}")
        
        if not self.folders and not self.trajectories and not self.ground_truth:
            messagebox.showwarning("警告", "没有找到有效的文件夹或轨迹文件")
            return
        
        max_folder = max((f.get_image_count() for f in self.folders), default=0)
        max_traj = max((max(t.frames) + 1 if t.frames else 0 for t in self.trajectories), default=0)
        max_gt = (max(self.ground_truth.frames) + 1) if self.ground_truth and self.ground_truth.frames else 0
        self.max_frames = max(max_folder, max_traj, max_gt)
        
        self._create_panels()
        self.frame_slider.config(to=max(0, self.max_frames - 1))
        self.current_frame = 0
        self._display_current_frame()
        
        gt_status = ", 有真值对比" if self.ground_truth else ""
        self.status_label.config(text=f"已加载 {len(self.folders)} 文件夹, {len(self.trajectories)} 轨迹{gt_status}, {self.max_frames} 帧")
    
    def _create_panels(self):
        for panel in self.panels:
            panel.destroy()
        self.panels.clear()
        
        if self.trajectory_panel:
            self.trajectory_panel.destroy()
            self.trajectory_panel = None
        
        # 当有轨迹或真值时显示轨迹面板
        has_traj_panel = self.trajectories or self.ground_truth
        
        total = len(self.folders) + (1 if has_traj_panel else 0)
        self.cols = math.ceil(math.sqrt(total))
        self.rows = math.ceil(total / self.cols)
        
        idx = 0
        for folder in self.folders:
            panel = ImagePanel(self.panel_container, folder, (self.panel_width, self.panel_height),
                              on_close=self._on_panel_close)
            panel.grid(row=idx // self.cols, column=idx % self.cols, padx=5, pady=5, sticky='nsew')
            self.panels.append(panel)
            idx += 1
        
        if has_traj_panel:
            self.trajectory_panel = TrajectoryPanel(self.panel_container, self.trajectories,
                                                    (self.panel_width, self.panel_height),
                                                    on_close=self._on_trajectory_panel_close,
                                                    ground_truth=self.ground_truth,
                                                    gt_frame_offset=self.gt_frame_offset)
            self.trajectory_panel.grid(row=idx // self.cols, column=idx % self.cols, padx=5, pady=5, sticky='nsew')
        
        for i in range(self.cols):
            self.panel_container.columnconfigure(i, weight=1)
        for i in range(self.rows):
            self.panel_container.rowconfigure(i, weight=1)
    
    def _on_panel_close(self, panel: ImagePanel):
        if panel in self.panels:
            self.panels.remove(panel)
            self.folders.remove(panel.folder)
            panel.destroy()
            self._relayout_panels() if self.panels or self.trajectory_panel else self._clear_all()
    
    def _on_trajectory_panel_close(self, panel: TrajectoryPanel):
        self.trajectory_panel.destroy()
        self.trajectory_panel = None
        self.trajectories.clear()
        self._relayout_panels() if self.panels else self._clear_all()
    
    def _relayout_panels(self):
        total = len(self.panels) + (1 if self.trajectory_panel else 0)
        self.cols = math.ceil(math.sqrt(total))
        
        idx = 0
        for panel in self.panels:
            panel.grid(row=idx // self.cols, column=idx % self.cols, padx=5, pady=5, sticky='nsew')
            idx += 1
        if self.trajectory_panel:
            self.trajectory_panel.grid(row=idx // self.cols, column=idx % self.cols, padx=5, pady=5, sticky='nsew')
        
        self._display_current_frame()
    
    def _display_current_frame(self):
        """同步显示当前帧"""
        for panel in self.panels:
            panel.display_frame(self.current_frame)
        if self.trajectory_panel:
            self.trajectory_panel.display_frame(self.current_frame)
        
        self.frame_label.config(text=f"{self.current_frame + 1}/{self.max_frames}")
        self.frame_var.set(self.current_frame)
    
    def _add_folder(self):
        folder = filedialog.askdirectory(title="选择图片文件夹")
        if folder:
            paths = [f.folder_path for f in self.folders] + [folder]
            trajs = [t.file_path for t in self.trajectories]
            self.load_data(paths, trajs)
    
    def _clear_all(self):
        self._stop_play()
        for panel in self.panels:
            panel.destroy()
        if self.trajectory_panel:
            self.trajectory_panel.destroy()
            self.trajectory_panel = None
        
        self.folders.clear()
        self.panels.clear()
        self.trajectories.clear()
        self.ground_truth = None  # 清除真值轨迹
        self.current_frame = self.max_frames = 0
        ImagePanel.pil_cache.clear()
        
        self.frame_slider.config(to=0)
        self.frame_label.config(text="0/0")
        self.status_label.config(text="就绪")
    
    def _reload_data(self):
        """重载数据 - 从配置文件或初始参数重新加载"""
        # 清除缓存
        ImagePanel.pil_cache.clear()
        
        # 重新加载配置
        if self.config_path and os.path.isfile(self.config_path):
            folders, trajectories, ground_truth, gt_offset = load_yaml_config(self.config_path)
            self.gt_frame_offset = gt_offset
            self.load_data(folders, trajectories, ground_truth)
            self.status_label.config(text=f"已重载配置: {os.path.basename(self.config_path)}")
        elif self.initial_folders or self.initial_trajectories:
            self.load_data(
                self.initial_folders or [],
                self.initial_trajectories or [],
                self.initial_ground_truth
            )
            self.status_label.config(text="已重载数据")
        else:
            messagebox.showinfo("提示", "没有可重载的配置")
    
    def _toggle_play(self):
        self._stop_play() if self.is_playing else self._start_play()
    
    def _start_play(self):
        if not self.folders and not self.trajectories:
            return
        self.is_playing = True
        self.play_btn.config(text="⏸")
        self._play_next()
    
    def _stop_play(self):
        self.is_playing = False
        self.play_btn.config(text="▶")
        if self.play_job:
            self.after_cancel(self.play_job)
            self.play_job = None
    
    def _play_next(self):
        if not self.is_playing or self.current_frame >= self.max_frames - 1:
            self._stop_play()
            return
        self.current_frame += 1
        self._display_current_frame()
        self.play_job = self.after(self.play_interval, self._play_next)
    
    def _next_frame(self):
        if self.current_frame < self.max_frames - 1:
            self.current_frame += 1
            self._display_current_frame()
    
    def _prev_frame(self):
        if self.current_frame > 0:
            self.current_frame -= 1
            self._display_current_frame()
    
    def _goto_first_frame(self):
        self.current_frame = 0
        self._display_current_frame()
    
    def _goto_last_frame(self):
        self.current_frame = max(0, self.max_frames - 1)
        self._display_current_frame()
    
    def _on_slider_change(self, value):
        new_frame = int(float(value))
        if new_frame != self.current_frame:
            self.current_frame = new_frame
            self._display_current_frame()
    
    def _on_speed_change(self, event):
        try:
            self.play_interval = max(10, 1000 // int(self.speed_var.get()))
        except ValueError:
            pass


def load_yaml_config(config_path: str) -> Tuple[List[str], List[str], Optional[str], int]:
    """加载 YAML 配置文件，返回 (folders, trajectories, ground_truth, gt_frame_offset)"""
    folders, trajectories = [], []
    ground_truth = None
    gt_frame_offset = 0
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config:
            # 读取文件夹列表
            if 'folders' in config and config['folders']:
                folders = config['folders'] if isinstance(config['folders'], list) else [config['folders']]
            
            # 读取轨迹文件列表 (支持 trajectory 或 trajectories)
            if 'trajectory' in config and config['trajectory']:
                traj = config['trajectory']
                trajectories = traj if isinstance(traj, list) else [traj]
            if 'trajectories' in config and config['trajectories']:
                traj = config['trajectories']
                trajectories.extend(traj if isinstance(traj, list) else [traj])
            
            # 读取真值
            if 'ground_truth' in config and config['ground_truth']:
                ground_truth = config['ground_truth']
            
            # 读取真值帧偏移量
            if 'ground_truth_frame_offset' in config:
                gt_frame_offset = int(config['ground_truth_frame_offset'])
    except Exception as e:
        print(f"加载配置文件失败 {config_path}: {e}")
    
    return folders, trajectories, ground_truth, gt_frame_offset


def main():
    parser = argparse.ArgumentParser(description="多文件夹图片播放器")
    parser.add_argument('-d', '--dirs', nargs='+', help='文件夹路径列表')
    parser.add_argument('-c', '--config', help='配置文件路径 (YAML)')
    parser.add_argument('-t', '--trajectories', nargs='+', help='轨迹文件路径')
    parser.add_argument('-g', '--ground-truth', help='真值文件路径')
    parser.add_argument('--gt-offset', type=int, default=0, help='真值帧偏移量')
    args = parser.parse_args()
    
    folders, trajectories, ground_truth = [], [], None
    gt_frame_offset = 0
    
    # 确定配置文件路径
    config_path = args.config
    if not config_path and not args.dirs:
        # 无命令行参数时使用默认配置
        if os.path.isfile(DEFAULT_CONFIG_PATH):
            config_path = DEFAULT_CONFIG_PATH
            print(f"使用默认配置: {DEFAULT_CONFIG_PATH}")
    
    # 加载配置
    if args.dirs:
        folders = args.dirs
    elif config_path:
        folders, trajectories, ground_truth, gt_frame_offset = load_yaml_config(config_path)
    
    if args.trajectories:
        trajectories.extend(args.trajectories)
    
    if args.ground_truth:
        ground_truth = args.ground_truth
    
    if args.gt_offset != 0:
        gt_frame_offset = args.gt_offset
    
    root = tk.Tk()
    root.title("Multi-Folder Image Player")
    root.geometry("1400x900")
    root.minsize(800, 600)
    
    ttk.Style().theme_use('clam')
    MultiFolderPlayer(root, folders or None, trajectories or None, ground_truth, gt_frame_offset, config_path)
    root.mainloop()


if __name__ == "__main__":
    main()
