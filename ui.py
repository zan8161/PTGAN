from pathlib import Path
import gc
import math
import numpy as np
import os.path as osp
import sys
from PIL import Image
import gradio as gr
import torch
from metrics import metrices_map
import os
from PIL import Image, ImageDraw, ImageFont, ImageOps
from config import cfg
from dataset import get_transform
from models import TransReIDBase_Inference
from tools.jsonio import read_json


class ReID_Pipline():
    
    def __init__(self, cfg, gallery_camIDs:list[int], gallery_precalculate_path:os.PathLike) -> None:
        
        self.dev = torch.device(f"cuda:{cfg.MODEL.DEVICE_ID}") if torch.cuda.is_available() else torch.device("cpu")

        self.model = self._build_model(cfg=cfg).to(device=self.dev)
        
        self.camera_wised_gallery_feature:list[torch.Tensor] = [
            torch.load(
                Path(gallery_precalculate_path)/f"c{i:03d}"/"reid_features.pt",
                weights_only=True, map_location="cpu"
            ).to(device=self.dev) 
            for i in gallery_camIDs
        ]

        self.camera_wised_gallery_item = [
            read_json(Path(gallery_precalculate_path)/f"c{i:03d}"/"items.json")
            for i in gallery_camIDs
        ]

        self.T = get_transform(
            cfg.INPUT.SIZE_TEST, 
            {'mean':cfg.INPUT.PIXEL_MEAN, 'std':cfg.INPUT.PIXEL_STD}
        )
    
    def _build_model(self, cfg) -> TransReIDBase_Inference:
        M = TransReIDBase_Inference(cfg=cfg)
        M.eval()
        return M

    def __call__(self, x:Image.Image, distance_method:str="cosine", topk:int=1, **kwargs) ->tuple[list[list[os.PathLike]], list[list[float]]]:
        
        xi:torch.Tensor = self.T(x)
        if xi.ndim == 3:
            xi = xi.unsqueeze(0) 
        qf = self.model(xi.to(self.dev))
        matched_filenames = [None]*len(self.camera_wised_gallery_feature)
        matched_score = [None]*len(self.camera_wised_gallery_feature)
        
        for ci in range(len(self.camera_wised_gallery_feature)):
            dist_matrix:torch.Tensor = metrices_map[distance_method](
                qf,self.camera_wised_gallery_feature[ci], 
                **kwargs
            )
            
            most_sims = dist_matrix.argsort(dim=1)[0][:topk]
        
            matched_filenames[ci] = [
                self.camera_wised_gallery_item[ci][_] 
                for _ in most_sims
            ]
            matched_score[ci] = [
                float(dist_matrix[0][_])
                for _ in most_sims
            ]

        return matched_filenames,  matched_score

reid_pipline:ReID_Pipline = None

class WebUI():
    
    def __init__(
            self, gallery_camIDs:list[int], show_size:tuple[int]=(80,80), 
            gallery_img_root:os.PathLike = Path("VehicleData")/"gallery",
            log_file:str=None
        ) -> None:
        self.log_fp = open(log_file, "w+") if log_file is not None else sys.stdout
        self.USE_FONT = None
        match sys.platform:
            case  "win32":
                self.USE_FONT = "C:\\Windows\\Fonts\\times.ttf" 
            case "linux":
                self.USE_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        self.g_root = gallery_img_root
        self.q_img = None
        self.q_bar = None
        self.show_size = show_size
        self.gallery_cameras = gallery_camIDs
        self.metrics_args_map = {
            're_ranking':{
                'k1':50, 
                'k2':15, 
                'lambda_value':0.3,
                'pbar':  gr.Progress()

            }
        }
        self.current_q:str= None
    
    def _get_query_img(self, bn:gr.File):
        self.current_q = bn.name
        im = Image.open(self.current_q).convert("RGB")
        return im
    
    def run_app(self):

        theme = gr.themes.Default(primary_hue="orange", text_size=gr.themes.sizes.text_lg).set(
            loader_color="#FF0000",slider_color="#FF0000"
        )

        with gr.Blocks(
            theme=theme, 
            css="""
                .gradio-container {background-color: black;}
                .scrollable { overflow-y: auto; height: 500px; }  /* Vertically scrollable */
            """, 
            title="Cross Camera Vehicle ReID Web UI", 
            fill_height=True, 
            # fill_width=True
        ) as demo:
    
            # Title Label for the Interface
            title = gr.Label("PTGAN - Cross Camera Vehicle ReID Demo", show_label=False)
            result_imgs1 = [None] * len(self.gallery_cameras)  # Placeholder for result images
            CL = [None]* len(self.gallery_cameras)  
            with gr.Row(equal_height=False):
                
                # Left Column for Query Image and Dropdown
                with gr.Column(scale=0.1):
                    """with gr.Column():
                        get_path_button = gr.Button("Get File Path")  # Button to trigger file path retrieval

                        file_input = gr.File(label="Upload a file", height=10)
                    """
                    with gr.Column():
                        self.q_bar = gr.Label(label="Query:", show_label=False)
                    
                    with gr.Column():
                        self.q_img = gr.Image(
                            label='Query', type='filepath', 
                            show_download_button=False
                        )
                   
                    with gr.Column():
                        metrices_bar = gr.Dropdown(
                            label="Distance Type:", 
                            choices=["cosine", "re_ranking"],
                            value="cosine"
                        )
                        reid_bnt = gr.Button("Search")  # Search Button
                    
                    with gr.Column():
                        topk_bar = gr.Dropdown(
                            label="topK", 
                            choices=[_+1 for _ in range(10)],
                            value=5
                        )
                
                with gr.Row(equal_height=True, elem_classes="scrollable"):  # Vertically scrollable column
                    with gr.Column():
                        for i, ci in enumerate(self.gallery_cameras):
                            CL[i] = gr.Label(f"camera {ci}", show_label=False, scale=0.2, color="#A4A4A4")
                            result_imgs1[i] = gr.Image(
                                label=f"",
                                show_download_button=False,
                                show_label=False,
                                container=True,  # This makes sure the image fills the block
                                height="auto",
                                scale=4
                            )
                
                reid_bnt.click(
                    self.Inference, 
                    inputs=[self.q_img, metrices_bar, topk_bar],
                    outputs= [self.q_bar] + result_imgs1
                )
                  
            demo.launch(share=True)
            demo.close()
    
    def write_log(self, result:tuple[list, list]):
        for cid in range(len(result[0])):
            print(f"{self.gallery_cameras[cid]} : ")
            for topk, (im, d) in enumerate(zip(result[0][cid], result[1][cid])):
                print(f"top{topk} : {im} (distance {d:.6f})", file=self.log_fp)
            print("="*20, file=self.log_fp)

    def Inference(self,query_path:str, metrics:str, topk:int):
        gc.collect()
        q_vid = self.extract_vid(query_path)
        print(f"query : {q_vid} ( {Path(query_path).stem} )", file=self.log_fp)
        q = Image.open(query_path).convert("RGB")
        additional_args = self.metrics_args_map.get(metrics, {})
        candidates, dist = reid_pipline(q, distance_method=metrics, topk=topk, **additional_args)
        self.write_log(result=(candidates, dist))
        candidates_imgs = [
            self.imcat(
                [
                    self.adding_text_card(
                        image_path = (self.g_root/f"c{ci+1:03d}"/cc),
                        txt=f"{dc:.3f}", topk=k+1,  
                        border_color=(255,0,0) \
                            if int(cc.split("_")[0]) == q_vid else (0,0,0)
                    )
                    for k, (cc, dc) in enumerate(zip(candidates[ci], dist[ci]))
                ]
            )
            for ci in range(len(candidates))
        ]
        return [f"Query CarID:{q_vid}"] + candidates_imgs 
    
    def imcat(self, imgs:list[Image.Image], line:int=5) -> Image.Image:
        canvas = None 
        if len(imgs) % line != 0:
            append = math.ceil(len(imgs)/line)*line - len(imgs)
            imgs += [np.zeros((imgs[0].size[1], imgs[0].size[0], 3), dtype=np.uint8) for _ in range(append)]
        
        for i in range(len(imgs)//line):

            L = np.hstack(
                [
                    np.array(img) if isinstance(img, Image.Image) else img 
                    for img in imgs[i*line: (i+1)*line] 
                ]
            )
            if canvas is None:
                canvas = L.copy()
            else:
                canvas = np.vstack([canvas, L])

        return Image.fromarray(canvas)

    def extract_vid(self, img_path:Path|os.PathLike) -> int:
        return int(Path(img_path).stem.split("_")[0])

    def adding_text_card(self, image_path:os.PathLike, topk:int, txt:str, border:int=5, border_color:tuple[int] = (100, 100, 100)) -> Image.Image:
  
        image = Image.open(image_path).resize((200,200))
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(self.USE_FONT, size=16)
        text_size = font.getbbox(txt)
        box_color =  (255, 165, 0)
        # Draw the text box
        draw.rectangle(
            xy=[(image.width - text_size[2] - 1 - 10 , 0), 
                (image.width-1, text_size[3] + 10 )], 
            fill=box_color
        )

        draw.text((image.width - text_size[2] - 5 ,  (text_size[3] + 10)/2 - 10) , txt, fill=(0,0,0), font=font)
        image = ImageOps.expand(image, border=border, fill=border_color)
        
        result_image_info =  Image.new('RGB', (image.size[0], image.size[1]//4), color=(0,0,0))
        draw_info = ImageDraw.Draw(result_image_info)
        info_font = ImageFont.truetype(self.USE_FONT, size=24)
        result_vid = f"CarID:{self.extract_vid(image_path)}"
        draw_info.text((result_image_info.size[0]//2 - 40, 10), result_vid, fill=(255,255,255), font=info_font)
        
        topk_label = Image.new('RGB', (image.size[0], image.size[1]//6), color=(0,0,0))
        topk_info = ImageDraw.Draw(topk_label)
        topk_font = ImageFont.truetype(self.USE_FONT, size=22)
        topk_info.text((topk_label.size[0]//2, 10), f"top{topk}", fill=(255,255,255), font=topk_font)

        image = np.vstack([np.array(topk_label), np.array(image),np.array(result_image_info)])
        
        return Image.fromarray(image)

if __name__ == "__main__":
    
    cfg.merge_from_file(Path("config")/"transreid_256_veri_gan.yml")
    cfg.freeze()
    
    precalculated_gallery = Path("VehicleData")/"precalculated"/"gallery"
    gallery_img_root = Path("VehicleData")/"camera_wised_gallery"
    gallery_camIDs = [_ for _ in range(1, 9)]

    print(f"Gallery : {precalculated_gallery} cameras {gallery_camIDs}")
    reid_pipline = ReID_Pipline(
        cfg=cfg,
        gallery_camIDs = gallery_camIDs,
        gallery_precalculate_path = precalculated_gallery
    )
    
    ui = WebUI(gallery_camIDs=gallery_camIDs, gallery_img_root = gallery_img_root)
    ui.run_app()
