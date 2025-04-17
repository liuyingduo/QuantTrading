import requests
import json
import time

def get_rzrq_data(date, max_pages=None):
    """
    获取指定日期的融资融券数据
    
    参数:
    date: str - 日期，格式如'2025-04-16'
    max_pages: int - 最大获取页数(可选)，None表示获取所有页
    """
    base_url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    headers = {
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Connection": "keep-alive",
        "Referer": "https://data.eastmoney.com/rzrq/detail/all.html",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
    }
    
    # 初始参数
    params = {
        "reportName": "RPTA_WEB_RZRQ_GGMX",
        "columns": "ALL",
        "source": "WEB",
        "pageSize": 50,  # 每页条数
        "sortColumns": "RZJME",
        "sortTypes": "-1",
        "filter": f"(DATE='{date}')",
        "_": int(time.time() * 1000)  # 时间戳
    }
    
    all_data = []
    page = 1
    total_pages = None
    
    while True:
        if max_pages and page > max_pages:
            break
            
        # 更新页码参数
        params.update({
            "pageNumber": page,
            "pageNo": page,
            "p": page,
            "pageNum": page,
            "_": int(time.time() * 1000)
        })
        
        try:
            response = requests.get(
                base_url,
                params=params,
                headers=headers,
                timeout=10
            )
            
            # 处理JSONP响应
            if response.text.startswith("datatable"):
                json_str = response.text.split("(", 1)[1].rsplit(")", 1)[0]
                data = json.loads(json_str)
            else:
                data = response.json()
                
            if not data.get("success"):
                print(f"请求失败: {data.get('message')}")
                break
                
            # 第一次请求获取总页数
            if total_pages is None:
                total_pages = data["result"]["pages"]
                print(f"共发现 {total_pages} 页数据，每页 {params['pageSize']} 条")
                
            # 添加数据
            page_data = data["result"]["data"]
            all_data.extend(page_data)
            print(f"已获取第 {page}/{total_pages} 页，本页 {len(page_data)} 条数据")
            
            # 检查是否已获取所有数据
            if page >= total_pages:
                break
                
            page += 1
            time.sleep(1)  # 适当延迟，避免请求过于频繁
            
        except Exception as e:
            print(f"获取第 {page} 页数据时出错: {str(e)}")
            break
            
    print(f"\n数据获取完成，共获取 {len(all_data)} 条记录")
    return all_data

def get_rzrq_sector_data():
    """获取融资融券行业板块数据"""
    print("正在获取融资融券行业板块数据...")
    
    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36 Edg/135.0.0.0',
        'Referer': 'https://data.eastmoney.com/rzrq/hy.html'
    }
    
    all_data = []
    page = 1
    
    while True:
        params = {
            'callback': f'datatable{int(time.time()*1000)}',
            'reportName': 'RPTA_WEB_BKJYMXN',
            'columns': 'ALL',
            'pageNumber': page,
            'pageNo': page,
            'pageSize': 50,
            'sortColumns': 'FIN_NETBUY_AMT',
            'sortTypes': '-1',
            'stat': '1',
            'filter': '(BOARD_TYPE_CODE="005")',
            'p': page,
            'pageNum': page,
            '_': int(time.time()*1000)
        }
        
        try:
            print(f"正在获取第 {page} 页数据...")
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            # 处理JSONP格式响应
            json_data = response.text
            json_data = json_data.split('(', 1)[1].rsplit(')', 1)[0]
            data = json.loads(json_data)
            
            # 检查是否有数据
            if data.get('result') and data['result'].get('data'):
                records = data['result']['data']
                all_data.extend(records)
                
                # 如果不足50条记录，说明已经是最后一页
                if len(records) < 50:
                    break
            else:
                break
                
            page += 1
            time.sleep(1)  # 适当延迟，避免请求过于频繁
            
        except Exception as e:
            print(f"获取第 {page} 页数据时出错: {str(e)}")
            break
            
    print(f"\n数据获取完成，共获取 {len(all_data)} 条记录")
    return all_data

# 使用示例
if __name__ == "__main__":
    # 获取2025-04-16的数据
    date = "2025-04-16"
    rzrq_data = get_rzrq_data(date)
    
    # 保存到JSON文件
    with open(f"\\data\\rzrq_data_{date}.json", "w", encoding="utf-8") as f:
        json.dump(rzrq_data, f, ensure_ascii=False, indent=2)
    
    # 打印前5条数据
    for i, item in enumerate(rzrq_data[:5]):
        print(f"\n第{i+1}条数据:")
        print(f"股票代码: {item['SCODE']}")
        print(f"股票名称: {item['SECNAME']}")
        print(f"融资余额: {item['RZYE']}")
        print(f"融券余量: {item['RQYL']}")
    
    # 获取行业板块融资融券数据
    sector_data = get_rzrq_sector_data()
    
    # 保存到JSON文件
    with open(f".\\data\\rzrq_sector_data.json", "w", encoding="utf-8") as f:
        json.dump(sector_data, f, ensure_ascii=False, indent=2)
    
    # 打印前5条行业板块数据
    for i, item in enumerate(sector_data[:5]):
        print(f"\n第{i+1}条行业板块数据:")
        print(f"板块名称: {item.get('BOARD_NAME', '')}")
        print(f"融资买入额: {item.get('FIN_BUY_AMT', '')}")
        print(f"融资偿还额: {item.get('FIN_REFUND_AMT', '')}")
        print(f"融资净买额: {item.get('FIN_NETBUY_AMT', '')}")