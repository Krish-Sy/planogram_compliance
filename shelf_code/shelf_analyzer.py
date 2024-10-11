import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
import openpyxl 
import cv2
import os
import io
import json
import requests
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
from dotenv import load_dotenv

load_dotenv()

class ShelfAnalyzer:
    def __init__(self, yolo_model_path, clip_model_name="openai/clip-vit-base-patch32"):
        self.yolo_model = YOLO(yolo_model_path)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    def detect_products(self, image):
        results = self.yolo_model(image)
        
        if results and len(results[0].boxes.xyxy) > 0:
            return results[0].boxes.xyxy.cpu().numpy()
        else:
            print("No objects detected in the image.")
            return []

    def calculate_midpoint(self, bbox):
        x_min, y_min, x_max, y_max = bbox
        return (x_min + x_max) / 2, (y_min + y_max) / 2

    def group_images_by_rows(self, bboxes):
        rows = []
        for bbox in bboxes:
            mid_x, mid_y = self.calculate_midpoint(bbox)
            added_to_row = False
            for row in rows:
                first_bbox = row[0]
                row_y_min, row_y_max = first_bbox[1], first_bbox[3]
                if row_y_min <= mid_y <= row_y_max:
                    row.append(bbox)
                    added_to_row = True
                    break
            if not added_to_row:
                rows.append([bbox])
        rows = sorted(rows, key=lambda row: min(bbox[1] for bbox in row))
        rows = [sorted(row, key=lambda x: x[0]) for row in rows]
        return rows

    def compute_color_histogram(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def compute_color_similarity(self, hist1, hist2):
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    def compute_clip_similarity(self, image1, image2):
        pil_image1 = Image.fromarray(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
        pil_image2 = Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
        inputs = self.clip_processor(
            images=[pil_image1, pil_image2],
            return_tensors="pt",
            padding=True
        )
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = torch.nn.functional.cosine_similarity(
                image_features[0].unsqueeze(0),
                image_features[1].unsqueeze(0)
            ).item()
        return similarity

    def compute_combined_similarity(self, image1, image2, clip_weight=0.7):
        clip_sim = self.compute_clip_similarity(image1, image2)
        hist1 = self.compute_color_histogram(image1)
        hist2 = self.compute_color_histogram(image2)
        color_sim = self.compute_color_similarity(hist1, hist2)
        combined_sim = (clip_sim * clip_weight) + (color_sim * (1 - clip_weight))
        return combined_sim

    def count_similar_products(self, row_images, similarity_threshold=0.8):
        if not row_images:
            return []
        current_group = [0]
        groups = []
        for i in range(1, len(row_images)):
            similarity = self.compute_combined_similarity(row_images[i-1], row_images[i])
            if similarity >= similarity_threshold:
                current_group.append(i)
            else:
                groups.append(current_group)
                current_group = [i]
        groups.append(current_group)
        return groups

    def google_lens_search(self, image):
        cloudinary.config(
            cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
            api_key=os.getenv("CLOUDINARY_API_KEY"),
            api_secret=os.getenv("CLOUDINARY_API_SECRET"),
            secure=True
        )

        _, buffer = cv2.imencode('.jpg', image)
        image_stream = io.BytesIO(buffer)
        response = cloudinary.uploader.upload(image_stream, resource_type='image')
        image_url = response['url']

        url = "https://www.searchapi.io/api/v1/search"
        params = {
            "engine": "google_lens",
            "url": image_url,
            "api_key": os.getenv("SEARCH_API_KEY") 
        }

        try:
            response = requests.get(url, params=params)
            response_data = response.json()

            if 'visual_matches' in response_data and len(response_data['visual_matches']) > 0:
                res = response_data['visual_matches'][0].get('title', 'No title found in visual matches')
            elif 'exact_matches' in response_data and len(response_data['exact_matches']) > 0:
                res = response_data['exact_matches'][0].get('title', 'No title found in exact matches')
            elif 'related_searches' in response_data and len(response_data['related_searches']) > 0:
                res = response_data['related_searches'][0].get('title', 'No title found in related searches')
            else:
                res = 'No matches found'
            
            print(res)
            return res

        except requests.RequestException as e:
            print(f"An error occurred with the API request: {e}")
            return 'API request error'
        except KeyError as e:
            print(f"Unexpected data format in response: {e}")
            return 'Data format error'

    def analyze_shelf(self, image_path):
        image = cv2.imread(image_path)
        bboxes = self.detect_products(image)
        rows = self.group_images_by_rows(bboxes)
        product_info = []

        for row_idx, row in enumerate(rows):
            row_images = [image[int(y1):int(y2), int(x1):int(x2)]
                        for x1, y1, x2, y2 in row]
            groups = self.count_similar_products(row_images)

            for group_idx, group in enumerate(groups):
                representative_image = row_images[group[0]]
                lens_result = self.google_lens_search(representative_image)
                group_bboxes = [row[i] for i in group]  

                # Format the bounding boxes as strings separated by a semicolon
                formatted_group_bboxes = ';'.join(
                    [','.join(map(str, map(float, row[i]))) for i in group]
                )

                # Representative bounding box
                formatted_bounding_box = ','.join(map(str, map(float, row[group[0]])))

                product_info.append({
                    "Bounding Box": formatted_bounding_box,  
                    "Group Bounding Boxes": formatted_group_bboxes, 
                    "Detected Count": len(group),
                    "Row": row_idx + 1,
                    "Google Lens Result": lens_result
                })

        return product_info


    def mark_group_in_image(self, image, group_bboxes, similarity_score, detected_count, expected_count):
        overlay = image.copy()
        alpha = 0.5  

        if similarity_score < 0.78:
            mismatch_text = "Product Mismatch"
        elif detected_count != expected_count:
            mismatch_text = "Count Mismatch"
        else:
            mismatch_text = ""

        if mismatch_text:
            for bbox in group_bboxes:
                bbox = bbox.astype(int)  
                x_min, y_min, x_max, y_max = bbox
                cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 0, 255), -1) 
            
            image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            
            x_min, y_min, _, _ = group_bboxes[0].astype(int)  
            cv2.putText(image_new, mismatch_text, (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 7)  
        else:
            image_new = image 
        
        return image_new


    def mark_image(self, image_path, product_info, excel_input_path, output_image_path):
        existing_df = pd.read_excel(excel_input_path)
        product_df = pd.DataFrame(product_info)

        image = cv2.imread(image_path)
        df = pd.concat([existing_df, product_df], axis=1)

        for index, row in df.iterrows():
            google_lens_result = row['Google Lens Result']

            representative_bbox = np.array([float(coord) for coord in row['Bounding Box'].split(',')])
            group_bboxes = [np.array([float(coord) for coord in bbox.split(',')]) 
                            for bbox in row['Group Bounding Boxes'].split(';')]

            detected_count = row['Detected Count']
            row_idx = index

            expected_count = row['Count']

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')

            sentence1 = str(row['Product_name'])
            sentence2 = str(row['Google Lens Result'])

            tokens1 = tokenizer.tokenize(sentence1)
            tokens2 = tokenizer.tokenize(sentence2)

            tokens_1 = ['[CLS]'] + tokens1 + ['[SEP]']
            tokens_2 = ['[CLS]'] + tokens2 + ['[SEP]']

            input_id1 = torch.tensor(tokenizer.convert_tokens_to_ids(tokens_1)).unsqueeze(0)
            input_id2 = torch.tensor(tokenizer.convert_tokens_to_ids(tokens_2)).unsqueeze(0)

            with torch.no_grad():
                outputs1 = model(input_id1)
                outputs2 = model(input_id2)
                embeddings1 = outputs1.last_hidden_state[:, 0, :]
                embeddings2 = outputs2.last_hidden_state[:, 0, :]
            
            similarity_score = cosine_similarity(embeddings1, embeddings2)[0][0]

            if similarity_score < 0.7 or detected_count != expected_count:
                image = self.mark_group_in_image(image, group_bboxes, similarity_score, detected_count, expected_count)

        cv2.imwrite(output_image_path, image)
        print(f"Marked image saved to {output_image_path}")

def main():
    try:
        yolo_model_path = "/Users/krish/Development/AI_ML/Planogram_demo/best.pt"
        image_path = "/Users/krish/Development/AI_ML/Planogram_demo/2d-planogram-model (6).png"
        excel_input_path = "/Users/krish/Development/AI_ML/Planogram_demo/2d-planogram-wrong-demo.xlsx"
        output_image_path = "/Users/krish/Development/AI_ML/Planogram_demo/marked_output.png"

        analyzer = ShelfAnalyzer(yolo_model_path)
        product_info = analyzer.analyze_shelf(image_path)
        analyzer.mark_image(image_path, product_info, excel_input_path, output_image_path)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

#Link Excel - download
#Add colors
#Use real images
#Add summary