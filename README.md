rtx 5090 laptop 
Pytorch 2.7.0+Cu128
python：3.11.11
natten：0.20.1

Cutlass modify:
![image](https://github.com/user-attachments/assets/534f569e-09c5-417d-8173-a205e74f705d)
![image](https://github.com/user-attachments/assets/05b76639-0a1f-4a07-969d-1436af7ee459)

Natten modify:
![image](https://github.com/user-attachments/assets/aea12456-c0f8-4258-8551-4b93a7d04e2b)
![image](https://github.com/user-attachments/assets/9a033bd9-6aab-4062-9171-5788ebdb9095)
![image](https://github.com/user-attachments/assets/8594404d-a6a2-4bff-9b53-74e6abdc1346)

注意：WindowsBuiler.bat 需要把你的Comfyui Python 地址放入其中
Notice：Windows Builer.bat needs to put your Comfyui Python address into it

ADD：
The comfyui-PMRF node code have some Problem！Must Modify!!!
问题是 PMRF竟然不支持新版0.20.1版本的Natten代码，需要修改
 Path：\your ComfyUI Folder\custom_nodes\ComfyUI-PMRF\arch\hourglass\image_transformer_v2.py

Search the code: if natten.has_fused_na():  modify to -------> if natten:   

  # if natten.has_fused_na():
        if natten:
            q, k, v = rearrange(qkv, "n h w (t nh e) -> t n h w nh e", t=3, e=self.d_head)
            q, k = scale_for_cosine_sim(q, k, self.scale[:, None], 1e-6)
            theta = self.pos_emb(pos)
            q = apply_rotary_emb_(q, theta)
            k = apply_rotary_emb_(k, theta)
            flops.op(flops.op_natten, q.shape, k.shape, v.shape, self.kernel_size)
            x = natten.functional.na2d(q, k, v, self.kernel_size, scale=1.0)
            x = rearrange(x, "n h w nh e -> n h w (nh e)")

This is the Picture to view：
<img width="1186" height="503" alt="Image" src="https://github.com/user-attachments/assets/ef64fcc7-f046-4a38-be76-3ad15df40113" />

All done,restart Comfyui, You will use PMRF node！！！

