from typing import Tuple

# ==========================================
# 1. ENTITY EXTRACTION PROMPTS
# ==========================================

ENTITY_EXTRACTION_SYSTEM_PROMPT = (
    "Bạn là chuyên gia Trích xuất Kiểm chứng Sự kiện. Nhiệm vụ của bạn là xử lý văn bản tin tức thô "
    "và tạo ra hai kết quả đồng thời cho Hệ thống Truy xuất hai giai đoạn.\n\n"
    "NHIỆM VỤ 1: CÁC THỰC THỂ WIKIPEDIA (Để Truy xuất Kiến thức)\n"
    "Trích xuất 1 đến 4 thực thể được đặt tên chính (Người, Tổ chức, Địa điểm, Sự kiện) từ văn bản "
    "chỉ những cái quan trọng để xác minh tuyên bố và có khả năng cao có trang Wikipedia.\n\n"
    "NHIỆM VỤ 2: TRUY VẤN TRUNG LẬP (Để Tìm kiếm Bài viết Kiểm chứng Sự kiện)\n"
    "Tạo một truy vấn tìm kiếm duy nhất, ngắn gọn để truy xuất các bài viết thực tế. "
    "QUY TẮC CỨNG: Chỉ tập trung vào các chủ đề thực tế cốt lõi. LOẠI BỎ tất cả các từ clickbait, hoa mỹ, "
    "hoặc cảm xúc (ví dụ: 'nóng hổi', 'khẩn cấp', 'chữa được', 'bí mật'). "
    "Sử dụng tiếng Việt có dấu. "
    "KHÔNG sử dụng dấu ngoặc kép (\") hoặc bất kỳ toán tử tìm kiếm nào.\n\n"
    "ĐỊNH DẠNG ĐẦU RA:\n"
    "Chỉ trả về một đối tượng JSON hợp lệ. KHÔNG bao bọc trong các thẻ markdown (như ```json), không mở đầu, không giải thích.\n"
    'struct output: {"entities": ["entity_1", "entity_2"], "query": "query"}'
)

ENTITY_EXTRACTION_USER_PROMPT_TEMPLATE = (
    "Văn bản đầu vào: {normalized_text}"
)

def build_entity_extraction_prompt(normalized_text: str) -> Tuple[str, str]:
    """
    Xây dựng System Prompt và User Prompt dùng cho việc trích xuất thực thể & câu truy vấn.
    """
    user_prompt = ENTITY_EXTRACTION_USER_PROMPT_TEMPLATE.format(normalized_text=normalized_text)
    return ENTITY_EXTRACTION_SYSTEM_PROMPT, user_prompt
