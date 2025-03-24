from Zhipu_gpt import zhipu_api

def get_setence_mood(sentence):
    system_prompt = '''
    请判断用户输入的句子输入什么四种情绪类别里面的哪一种，即平静、快乐、警觉和善良。如果用户输入的句子无明显情绪，那么输出平静；如果用户输入的句子是积极的，那么输出快乐；如果用户输入的句子是消极的，那么输出警觉；如果用户输入的句子是中性的，那么输出善良。
    输出：平静/快乐/警觉/善良。
    
    '''
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sentence},
    ]
    response = zhipu_api(messages)
    print(response)

if __name__ == "__main__":
    sentence = '''
    25年第11周（3.10-3.16）新能源排行榜：

1、比亚迪：6.28万，新能源市占率28.4%；

2、五菱：1.58万，新能源市占率7.1%；

3、特斯拉：1.53万，新能源市占率6.9%；

4、吉利：1.15万，新能源市占率5.2%；

5、理想：0.79万，新能源市占率3.6%。

其他，领克销量0.26万，超越问界和智界进入榜单第15名
    '''
    get_setence_mood(sentence)
