# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import fastdeploy as fd
import cv2
import requests
import base64

from http import HTTPStatus
import requests
import json
import time
import os


SHOW_VIDEO = False
llm_url = "http://127.0.0.1:8002/v1/chat/completions" 
# llm_url = "http://112.48.30.69:20240/v1/chat/completions" 
#prompt_template = 
'''
请按照从左到有，从上到下的顺序扫描整张图片，对以下11种要素进行判断：
1.有人未佩戴安全帽（特殊情况说明：若仅佩戴普通鸭舌帽、环绕式帽檐帽、厨师高帽等非安全帽，也算未佩戴。若是人物头部被遮挡或者头部不在/不完全在图中，则忽略该人物）； 
2.有人佩戴安全帽（需满足：安全帽必须完全覆盖头部，手持安全帽或者放置状态的安全帽不算佩戴安全帽（安全帽必须佩戴状态）；普通的鸭舌帽、礼帽、环绕式帽檐帽、厨师高帽等帽子不是安全帽，若只佩戴这些帽子则不算作佩戴安全帽。）；
3.地面上存在散落或裸露的电线（需满足：必须是电线，且电线必须和地面有接触。注意当检测到电线时先判断其是否在地上而非是悬挂在半空中或者在墙上。请注意甄别检测的物体，不要将电线和水管或者其他管类混淆。）；
4.未做交叉加固的施工脚手架，梯子除外；
5.下方存在未做足够防护的悬空区域；
6.木质折叠梯（需满足以下特征：表面为哑光、具有自然木纹理，颜色为原木色或涂色木面，通常不反光；结构连接多为钉子或绳子限位。若边缘弯曲圆滑，表面无明显金属反光，也倾向为木质。若存在金属质感、焊接、光泽、螺丝等，请不要判断为木质；若难以分辨材质，请选择否）；
7.金属折叠梯（需满足以下特征：表面呈银灰色或其他亮漆色，可能因使用而有污渍或刮痕；连接处常使用螺丝或铆钉，限位方式多为金属片或连接杆；边缘锐利，有反光光泽或金属质感。若表面哑光、颜色偏木头或带木纹，请不要判断为金属梯）；
8.灭火器；
9.人员吸烟；
10.人员动火作业；
11.人员在高楼层外部作业。
仅回答"是"或"否"，不要有任何别的回答，格式示例：
1.否\n2.否\n3.否\n4.否\n5.否\n6.否\n7.否\n8.否\n9.否\n10.否\n11.否
'''
# prompt_template = '请判断图上文字是横幅还是字幕,如果是布制横幅,则识别图片中的所有文字,合并为一行并以json格式返回,格式如下:{"banner":"文字"}.\n如果是其他,则返回"无".\n'

prompt_template1 = '''
请判断图片中是否存在道路积水，并输出判断结果和理由。道路积水包括但不限于：道路上有明显的水坑、积水区域、整条道路被水覆盖，或局部区域出现水面反光、水波纹、轮胎涉水、地面湿滑、行人绕行、车辆或物体部分浸泡在水中等现象。请特别关注局部区域的细节变化，如小范围的水洼、边角或低洼处的反光、积水痕迹，即使积水面积较小也要判断。判断时请结合水的颜色、反光、波纹、周围环境的变化等多种线索，避免遗漏局部积水。
请注意：不要将道路损坏的裂缝、坑洞、沥青色差、阴影、正常的黑白照片光影等误判为积水。只有在出现明显的水面反光、水波纹、轮胎涉水、地面湿滑等与水相关的特征时，才判断为积水。
若存在积水，请输出积水的判断结果、积水位置（如“左下角”“中间偏右”等）、积水范围大小（如“小范围”“大面积覆盖”等）等信息，并在理由中详细说明判断依据。
请仅以以下 JSON 格式输出结果，不要输出任何其他的解释说明或内容：
{
    "reason": "判断理由",
    "has_water_accumulation": true/false
}
请勿输出任何其他内容，仅输出上述 JSON。
'''

prompt_template2 = '''
请判断图片中是否存在违规的、经营性的占道经营行为，并输出判断结果及理由。占道经营仅指摊贩或商铺在非经营区域（如人行道、车行道，包含店门口的道路）摆放商品、桌椅、搭建棚子、设置摊位、售卖物品等与经营活动直接相关的行为,并且有人在从事经营活动，造成阻碍通行等情形。请勿将正常的公共设施（如垃圾桶、路灯、交通标志、环卫工具等）或临时停放的非经营性物品判断为占道经营,张贴及广告牌等不占用空间的物体也不算。请简要描述判断依据，例如：人行道上摆放了多个水果摊。
请仅以以下 JSON 格式输出结果，不要输出任何其他的解释说明或内容：
{
    "reason": "判断理由",
    "has_roadside_occupation": true/false
}
请勿输出任何其他内容，仅输出上述 JSON。
'''

key_word2 = "has_roadside_occupation"
key_word1 = "has_water_accumulation"

message_template=   [
                        {
                            "role": "user", 
                            "content": [
                                {
                                    "type": "text",
                                    "text": ""
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": None
                                    }
                                }
                            ]
                        }
                    ]

data_template = {
"model": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
"messages": [],
# "max_tokens": 12000,
"presence_penalty": 1.03,
"frequency_penalty": 1.0,
"seed": None,
"temperature": 0.1,
"top_p": 0.9,
"top_k": 2,
"stream": False
}

file_all = 0
file_true = 0
file_false = 0
file_error = 0

#遍历/public/lcy/xiaosan/dataset/dataset_doubao_1_5/dataset_wood_2，将jpg文件列表存为list
# path = "/public/lcy/xiaosan/dataset/dataset_doubao_1_5/dataset_wood_2/"
path = [
        # "/public/dataset/roadwater/"
        # "/public/dataset/stall/"
        # "/public/dataset/negative"
        "/public/dataset/negative/true_stall"
       ]
# 遍历包含子目录，收集所有图片文件
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
files = []
for single_path in path:
    for root, dirs, filenames in os.walk(single_path):
        for filename in filenames:
            if filename.lower().endswith(image_extensions):

                # files.append(os.path.relpath(os.path.join(root, filename), path))
                files.append(os.path.join(root, filename))
                file_all += 1

for file in files:
    print(file)

    filepath = file
    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist.")
        file_error += 1
        print("file processed: ", file_error + file_true + file_false, " / ", file_all)
        continue
    # 使用bytearray读取filepath直接转换为base64
    img_base64 = None
    # with open(filepath, "rb") as image_file:
    #     byte_array = image_file.read()
    #     img_base64 = base64.b64encode(byte_array).decode('utf-8')
    im = cv2.imread(filepath)
    if im is None:
        continue
    # 如果宽大于1024，则按比例缩放到宽为1024
    # max_width = 1024
    # height, width = im.shape[:2]
    # if width > max_width:
    #     scale = max_width / width
    #     new_width = max_width
    #     new_height = int(height * scale)
    #     im = cv2.resize(im, (new_width, new_height))
    ret, buffer_img = cv2.imencode('.jpg', im)    
    if ret:
        img_base64 = base64.b64encode(buffer_img).decode('utf-8')
    else:
        img_base64 = None
        print("image encode error")
        continue

    #读取同名txt文件
    answer = ""
    # txt_path = filepath.replace('.jpg', '.txt')
    # if os.path.exists(txt_path):
    #     with open(txt_path, "r") as txt_file:
    #         answer = txt_file.read()
    
    
    message = message_template

    # message[0]["content"][0]["text"] = prompt_template1
    message[0]["content"][1]["image_url"]["url"] = f"data:image/jpeg;base64,{img_base64}"
    #print(message)
    data = data_template
    data["messages"] = message
    # print(data)

    #打开output.txt准备持续写入结果
    output_file_water = open("output_water.txt", "a", encoding="utf-8")
    output_file_stall = open("output_stall.txt", "a", encoding="utf-8")
    
    prompts = [prompt_template1, prompt_template2]
    key_words = [key_word1, key_word2]
    output_files = [output_file_water, output_file_stall]

    for category in [1]:
        data["messages"][0]["content"][0]["text"] = prompts[category]
        # 发送请求到模型并接收响应
        start = time.time()
        try:
            with requests.post(llm_url, json=data, stream=False) as response:
                if response.status_code == HTTPStatus.OK:
                    content = json.loads(response.text)
                    content = content["choices"][0]["message"]["content"]
                    # print(content)
                    if content[-3] == '`':
                       lines = content.split('```')
                       # print(lines)
                       for line in lines:
                           if len(line) > 0:
                               if line[0] == 'j':
                                   jsonstr = line[5:-1]
                                   content = jsonstr
                                   break

                    print(content)
                    # content = content[8:-4]
                    words = json.loads(content)[key_words[category]]
                    reason = json.loads(content)["reason"]
                    # print(words)
                    write = filepath
                    if words == True:
                        write = "True \t" + write + "\t" + reason
                        file_true += 1
                    else:
                        write = "False\t" + write + "\t" + reason
                        file_false += 1

                    print("file processed: ", file_error + file_true + file_false, " / ", file_all)
                    output_files[category].write(f"{write}\n")
                    output_files[category].flush()

                else:
                    print('Status code: ', response.status_code,' error message: ', response.reason)
        except Exception as e:
            print(e)

        print("time: ", time.time()-start)
        # cv2.imshow("pic", im)
        # cv2.waitKey(5000)
