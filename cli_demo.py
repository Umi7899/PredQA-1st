import os
import platform
import signal
from transformers import AutoTokenizer, AutoModel
import readline
import lottery_predicter_pl_end

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True).half().cuda()
model = model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def build_prompt(history):
    prompt = "欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM-6B：{response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def main():
    # history = []
    global stop_stream
    prediction = lottery_predicter_pl_end.unique_cc_str
    print("欢迎使用预测模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        print("外侧循环\n")
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print("欢迎使用预测模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        current_length = 0
        new_query = """现在，请你扮演预测模型。已知根据AI模型在历史开奖数据上的分析，预测得到本周游戏“万能七码”可能的开奖结果如下：

        {pred}

        请你根据开奖结果回答用户的问题，用户的问题是：{question}"""
        new_query = new_query.replace("{question}", query).replace("{pred}", prediction)
        for response, history in model.stream_chat(tokenizer, new_query, history=[]):
            # print("内侧循环\n")
            if stop_stream:
                stop_stream = False
                break
            else:
                print(response[current_length:], end="", flush=True)
                current_length = len(response)
        # os.system(clear_command)
        # print(build_prompt(history), flush=True)


if __name__ == "__main__":
    main()
