# import time

# from ddgs import DDGS



# def test_bing_logic():

#     query = "Xăng E10"

#     print(f"🚀 [TESTING] Đang gọi DDGS với logic của ông...")

#     print(f"Backend: 'bing', Query: '{query}'")

    

#     start_time = time.time()

    

#     try:

#         # Giữ nguyên cấu trúc `with DDGS(timeout=20) as ddgs`

#         with DDGS(timeout=20) as ddgs:

#             # Ép cứng backend='bing' như code chính của ông

#             results_gen = ddgs.news(

#                 query=query,

#                 safesearch="off",

#                 timelimit=None,

#                 max_results=3,

#                 backend="bing"

#             )

            

#             # Chuyển thành list để lấy dữ liệu

#             results = list(results_gen)

            

#         duration = time.time() - start_time

#         print(f"✅ THÀNH CÔNG! Thời gian: {duration:.4f}s")

#         print(f"👉 Kết quả: {len(results)} items")

#         for r in results:

#             print(f"   -> {r.get('title')}")

            

#     except Exception as e:

#         print(f"❌ THẤT BẠI: {type(e).__name__} - {e}")



# if __name__ == "__main__":

#     test_bing_logic()

import httpx

def test_pure_request():
    url = "https://www.bing.com/news/search?q=X%C4%83ng+E10"
    print("🚀 Đang gửi request thuần (httpx)...")
    
    try:
        # Request không có bất kỳ giả lập nào
        response = httpx.get(url, timeout=10.0)
        print(f"✅ STATUS: {response.status_code}")
        print("Đã thông kết nối!")
    except Exception as e:
        print(f"❌ LỖI: {type(e).__name__}")
        print(f"Chi tiết: {e}")

if __name__ == "__main__":
    test_pure_request()