from ultralytics import YOLO
import os
import torch
import cv2 # 导入opencv
import math # 导入math模块用于角度计算
import argparse # 导入argparse模块用于命令行参数解析

class DigitPredictor:
    def __init__(self, ckpt_path="models/best.pt", imgsz=(256, 256), debug=True):
        self.model = YOLO(ckpt_path)
        self.imgsz = imgsz
        self.debug = debug
        # 默认分组参数
        self.grouping_params = {
            'min_vertical_overlap': 0.5,  # 最小垂直重叠率
            'max_horizontal_gap': 0.8,    # 最大水平间距比例
            'min_size_ratio': 0.7,        # 最小大小比例
            'distance_threshold_factor': 1.5, # 欧式距离阈值因子，乘以平均尺寸
        }
    
    def set_grouping_params(self, **params):
        """
        动态设置数字分组参数
        
        参数:
            min_vertical_overlap (float): 最小垂直重叠率 (0-1)
            max_horizontal_gap (float): 最大水平间距比例
            min_size_ratio (float): 最小大小比例 (0-1)
            distance_threshold_factor (float): 欧式距离阈值因子 (例如 2.0 表示 2 倍平均尺寸)
        """
        valid_params = {
            'min_vertical_overlap': (0.0, 1.0),
            'max_horizontal_gap': (0.0, 2.0),
            'min_size_ratio': (0.0, 1.0),
            'distance_threshold_factor': (0.1, 10.0) # 距离阈值因子范围
        }
        
        for param, value in params.items():
            if param in valid_params:
                min_val, max_val = valid_params[param]
                if min_val <= value <= max_val:
                    self.grouping_params[param] = value
                    if self.debug:
                        print(f"Updated {param}: {value}")
                else:
                    print(f"Warning: {param} value {value} out of range [{min_val}, {max_val}]")
            else:
                print(f"Warning: Unknown parameter {param}")

    def predictSingleDigit(self, source, conf_threshold=0.25, save_results=True):
        """
        预测单数字。
        Args:
            source: 图片路径或图片目录。
            conf_threshold: 置信度阈值。
            save_results: 是否保存预测结果图片。
        Returns:
            字典，键为图片路径，值为识别到的单数字（字符串形式）。
        """
        results = self.model.predict(source, save=save_results, imgsz=self.imgsz, conf=conf_threshold)
        
        recognized_single_digits = {}
        for result in results:
            path = result.path
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                # 对于单数字识别，我们只取第一个检测到的数字
                cls = int(boxes.cls[0].item())
                recognized_single_digits[path] = str(cls)
            else:
                recognized_single_digits[path] = "N/A" # 未检测到数字
        return recognized_single_digits

    def get_vector_angle(self, detection):
        """
        计算矩形框的方向向量角度。
        向量定义为从矩形中心点向左，并根据纵横比引入微小垂直分量。
        """
        width = detection['x2'] - detection['x1']
        height = detection['y2'] - detection['y1']
        
        # 向量的x分量为负（向左）
        vec_x = -width / 2
        
        # 向量的y分量，根据纵横比调整，模拟倾斜
        # 如果 width > height (矮胖)，y_vec 为负，向量偏向左下
        # 如果 height > width (高瘦)，y_vec 为正，向量偏向左上
        # 如果 width == height (正方形)，y_vec 为 0，向量纯粹向左
        vec_y = (height / 2) * (1 - width / height)
        
        # 计算角度（弧度）
        angle = math.atan2(vec_y, vec_x)
        # 转换为角度
        angle_deg = math.degrees(angle)
        return angle_deg
    
    def are_vectors_aligned(self, det1, det2, max_angle_diff=10):
        """检查两个矩形框的方向向量是否对齐"""
        angle1 = self.get_vector_angle(det1)
        angle2 = self.get_vector_angle(det2)
        
        # 计算角度差的绝对值
        angle_diff = abs(angle1 - angle2)
        # 如果角度差大于180度，取补角
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
            
        return angle_diff <= max_angle_diff
    
    def _group_detections(self, detections):
        """
        将检测到的数字按欧式距离初步分组，然后按方向向量进一步筛选。
        """
        if not detections:
            return []

        # 计算所有检测框的平均宽度，用于空间距离阈值
        avg_size = sum(d['x2'] - d['x1'] for d in detections) / len(detections)
        distance_threshold = avg_size * self.grouping_params['distance_threshold_factor']

        def get_center(det):
            """获取检测框的中心点"""
            return ((det['x1'] + det['x2']) / 2, (det['y1'] + det['y2']) / 2)

        def calc_distance(det1, det2):
            """计算两个检测框中心点之间的欧式距离"""
            c1 = get_center(det1)
            c2 = get_center(det2)
            return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

        # 第一阶段：基于欧式距离进行初步分组 (连通分量)
        initial_groups = []
        used_for_initial_grouping = set()

        if self.debug:
            print("\n第一阶段：欧式距离分组")
            print(f"检测框总数: {len(detections)}")
            print(f"平均宽度: {avg_size:.2f}")
            print(f"距离阈值: {distance_threshold:.2f} (因子: {self.grouping_params['distance_threshold_factor']})")

        for i, det in enumerate(detections):
            if i in used_for_initial_grouping:
                continue

            current_initial_group = []
            queue = [i]
            used_for_initial_grouping.add(i)

            while queue:
                idx = queue.pop(0)
                current_initial_group.append(detections[idx])

                for j, other_det in enumerate(detections):
                    if j not in used_for_initial_grouping:
                        dist = calc_distance(detections[idx], other_det)
                        dist_satisfied = dist < distance_threshold

                        if dist_satisfied:
                            queue.append(j)
                            used_for_initial_grouping.add(j)
            
            initial_groups.append(current_initial_group)
            if self.debug and current_initial_group:
                print(f"\n分组 #{len(initial_groups)} (包含 {len(current_initial_group)} 个检测框):")
                for idx, det in enumerate(current_initial_group):
                    center = ((det['x1'] + det['x2']) / 2, (det['y1'] + det['y2']) / 2)
                    print(f"  [{idx}] 数字: {det['cls']}")
                    print(f"      坐标: ({det['x1']:.1f},{det['y1']:.1f})-({det['x2']:.1f},{det['y2']:.1f})")
                    print(f"      中心点: ({center[0]:.1f}, {center[1]:.1f})")
                    print(f"      置信度: {det['conf']:.2f}")

        if self.debug:
            print("\n=== 第一阶段分组结果 ===")
            print(f"* 总分组数: {len(initial_groups)}")
            print(f"* 已分组检测框: {len(used_for_initial_grouping)}")
            print(f"* 未分组检测框: {len(detections) - len(used_for_initial_grouping)}")
            print("=" * 60)
        
        # 第二阶段：在每个初步分组内，根据向量对齐进一步分组
        final_groups = []
        for initial_group in initial_groups:
            if not initial_group:
                continue
            
            if self.debug:
                print(f"\nDebug (Stage 2): Processing initial group: {[d['cls'] for d in initial_group]}")

            subgroup_used = set()
            for i, det_in_group in enumerate(initial_group):
                if i in subgroup_used:
                    continue

                current_subgroup = [det_in_group]
                subgroup_used.add(i)

                for j, other_det_in_group in enumerate(initial_group):
                    if j not in subgroup_used:
                        is_aligned = self.are_vectors_aligned(det_in_group, other_det_in_group)
                        
                        if self.debug:
                            angle1 = self.get_vector_angle(det_in_group)
                            angle2 = self.get_vector_angle(other_det_in_group)
                            angle_diff = abs(angle1 - angle2)
                            if angle_diff > 180:
                                angle_diff = 360 - angle_diff
                            
                            dist = calc_distance(det_in_group, other_det_in_group)
                            dist_satisfied = dist < distance_threshold # 欧式距离在第一阶段已满足，这里仅为调试显示
                            
                            print(f"  Debug (Stage 2): Checking alignment between {det_in_group['cls']} (angle: {angle1:.2f}, pos: ({det_in_group['x1']:.1f},{det_in_group['y1']:.1f})) and {other_det_in_group['cls']} (angle: {angle2:.2f}, pos: ({other_det_in_group['x1']:.1f},{other_det_in_group['y1']:.1f}))")
                            print(f"    Angle Diff: {angle_diff:.2f}. Aligned: {is_aligned}")
                            print(f"    Distance: {dist:.2f} (Threshold: {distance_threshold:.2f}). Satisfied: {dist_satisfied}")

                        if is_aligned:
                            current_subgroup.append(other_det_in_group)
                            subgroup_used.add(j)
                
                # 对子组按x坐标排序
                final_groups.append(sorted(current_subgroup, key=lambda d: d['x1']))
        
        return final_groups

    def draw_detection_results(self, img, grouped_numbers):
        """
        在图像上绘制检测结果，包括边界框和数字标签
        
        Args:
            img: 原始图像
            grouped_numbers: 分组后的检测结果列表
            
        Returns:
            处理后的图像
        """
        for group in grouped_numbers:
            if not group:
                continue

            # 拼接数字
            number_str = "".join([str(d['cls']) for d in group])

            # 计算当前组的最小外接矩形
            min_x = min(d['x1'] for d in group)
            min_y = min(d['y1'] for d in group)
            max_x = max(d['x2'] for d in group)
            max_y = max(d['y2'] for d in group)

            # 绘制整体边界框
            color = (0, 255, 0)  # 绿色
            thickness = 2
            cv2.rectangle(img, (int(min_x), int(min_y)), (int(max_x), int(max_y)), color, thickness)

            # 添加识别到的数字文本
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            text_size = cv2.getTextSize(number_str, font, font_scale, font_thickness)[0]
            
            # 文本位置在框的左上角
            text_x = int(min_x)
            text_y = int(min_y) - 10 if int(min_y) - 10 > 10 else int(min_y) + text_size[1] + 10
            
            cv2.putText(img, number_str, (text_x, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)

        return img

    def predictMultipleDigits(self, source, conf_threshold=0.25, save_results=True):
        """
        预测多位数字。
        Args:
            source: 图片路径或图片目录。
            conf_threshold: 置信度阈值。
            save_results: 是否保存预测结果图片。
        Returns:
            字典，键为图片路径，值为识别到的多位数字（字符串形式）。
        """
        # 不让YOLOv8自动保存，我们手动处理
        results = self.model.predict(source, save=False, imgsz=self.imgsz, conf=conf_threshold)
        
        recognized_multi_digits = {}
        output_dir = "runs/detect/predict_multi_digits"
        os.makedirs(output_dir, exist_ok=True)

        for result in results:
            path = result.path
            boxes = result.boxes
            
            img = cv2.imread(path)
            if img is None:
                print(f"Warning: Could not read image {path}. Skipping.")
                continue

            detections = []
            if boxes is not None:
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    conf = boxes.conf[i].item()
                    cls = int(boxes.cls[i].item())
                    detections.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'conf': conf, 'cls': cls})

            # 使用新的两阶段分组逻辑
            grouped_numbers = self._group_detections(detections)

            recognized_multi_digits[path] = [] # 存储每个图片识别到的所有多位数字

            if save_results:
                # 绘制检测结果
                img = self.draw_detection_results(img, grouped_numbers)
                # 保存处理后的图片
                output_path = os.path.join(output_dir, os.path.basename(path))
                cv2.imwrite(output_path, img)
                print(f"Saved processed image to {output_path}")
            
            # 无论是否保存图片，都需要拼接数字字符串
            for group in grouped_numbers:
                if group:
                    number_str = "".join([str(d['cls']) for d in group])
                    recognized_multi_digits[path].append(number_str)

        return recognized_multi_digits

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='数字识别预测工具')
    parser.add_argument('--image', '-i', type=str, required=True, 
                       help='待预测的图片路径')
    parser.add_argument('--model', '-m', type=str, default='models/best.pt',
                       help='模型文件路径 (默认: models/best.pt)')
    parser.add_argument('--imgsz', type=int, nargs=2, default=[256, 256],
                       help='输入图像尺寸 [height width] (默认: 256 256)')
    parser.add_argument('--mode', type=str, choices=['single', 'multi'], default='single',
                       help='预测模式: single(单数字) 或 multi(多数字) (默认: single)')
    parser.add_argument('--save', action='store_true',
                       help='是否保存预测结果图片')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='置信度阈值 (默认: 0.25)')
    
    args = parser.parse_args()
    
    # 检查图片文件是否存在
    if not os.path.exists(args.image):
        print(f"错误: 图片文件 '{args.image}' 不存在!")
        exit(1)
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        print(f"错误: 模型文件 '{args.model}' 不存在!")
        exit(1)
    
    # 初始化预测器
    IMGSZ = tuple(args.imgsz)
    predictor = DigitPredictor(args.model, IMGSZ)
    
    print(f"使用模型: {args.model}")
    print(f"图像尺寸: {IMGSZ}")
    print(f"预测图片: {args.image}")
    print(f"预测模式: {args.mode}")
    print(f"置信度阈值: {args.conf}")
    print("-" * 50)
    
    if args.mode == 'single':
        # 单数字预测
        print("--- 单数字预测 ---")
        results = predictor.predictSingleDigit(args.image, conf_threshold=args.conf, save_results=args.save)
        for img_path, digit in results.items():
            print(f"图片: {os.path.basename(img_path)}, 识别结果: {digit}")
    
    elif args.mode == 'multi':
        # 多数字预测
        print("--- 多数字预测 ---")
        results = predictor.predictMultipleDigits(args.image, conf_threshold=args.conf, save_results=args.save)
        for img_path, digits in results.items():
            print(f"图片: {os.path.basename(img_path)}, 识别结果: {digits}")
    
    print("-" * 50)
    print("预测完成!")
    if args.save:
        print("结果已保存到 runs/detect/ 目录")