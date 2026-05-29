from typing import Tuple, Dict, List

# ==========================================
# 2. FINAL CLASSIFICATION PROMPTS
# ==========================================

FINAL_CLASSIFICATION_SYSTEM_PROMPT = (
    "Bạn là chuyên gia kiểm chứng tin tức. "
    "Nhiệm vụ: Phân loại bài viết dựa trên <VERIFIED_REPORTS> và <ENTITY_DEFINITIONS>.\n\n"
    "QUY TẮC:\n"
    "- Nếu dữ liệu RAG hỗ trợ nội dung -> 'Thật'.\n"
    "- Nếu dữ liệu RAG phản đối -> 'Giả'.\n"
    "- Nếu dữ liệu RAG lạc đề/thiếu thông tin -> LỜ ĐI, tự dự đoán dựa trên kiến thức của bạn.\n"
    "- Định dạng JSON: {\"label\": \"Thật\" hoặc \"Giả\", \"explanation\": \"Kết luận trước, sau đó giải thích ngắn gọn bằng chứng/lý do.\"}"
)

FINAL_CLASSIFICATION_USER_PROMPT_TEMPLATE = """NỀN TẢNG THÔNG TIN KIỂM CHỨNG:
<VERIFIED_REPORTS>
{rag_text}
</VERIFIED_REPORTS>
<ENTITY_DEFINITIONS>
{wiki_text}
</ENTITY_DEFINITIONS>

BÀI VIẾT CẦN PHÂN LOẠI:
Nội dung: "{text_input}"

VÍ DỤ MẪU:
{demo_text}

HƯỚNG DẪN:
- Phân tích và đưa ra JSON.
- Explanation PHẢI nêu kết luận ngay ở đầu câu.
- Nếu thông tin RAG lạc đề, hãy tự dự đoán dựa trên kiến thức của bạn và giải thích dự đoán.
- NHẤT ĐỊNH PHẢI TRẢ VỀ JSON HỢP LỆ, KHÔNG GIẢI THÍCH BÊN NGOÀI, KHÔNG DÙNG THẺ MARKDOWN.
- NHÃN PHẢI LÀ "Thật" hoặc "Giả", KHÔNG DÙNG TỪ NÀO KHÁC.
Kết luận (JSON):"""


def build_final_classification_prompt(
    text_input: str, 
    wiki_definitions: Dict[str, str], 
    rag_evidence: List[Dict], 
    fewshot_demos: List[Dict]
) -> Tuple[str, str]:
    """
    Xây dựng System Prompt và User Prompt phục vụ cho quá trình kiểm duyệt / kết luận tin tức cuối cùng.
    """
    # 1. Định nghĩa thực thể Wikipedia
    if wiki_definitions:
        wiki_text = "\n".join([f"- Entity: {k}\n  Definition: {v}" for k, v in wiki_definitions.items()])
    else:
        wiki_text = "- Entity: N/A\n  Definition: No entity definitions found."

    # 2. Ngữ liệu báo chí đối chiếu
    if rag_evidence:
        rag_text = ""
        for item in rag_evidence:
            rag_text += f"- Title: {item['title']}\n  Key Information: {item['chunk_text']}\n\n"
        rag_text = rag_text.strip()
    else:
        rag_text = "- Title: No verified report found\n  Key Information: No trusted fact evidence found."

    # 3. Ví dụ mẫu (Few-shot)
    if fewshot_demos:
        demo_text = ""
        for i, demo in enumerate(fewshot_demos, start=1):
            # Sử dụng trực tiếp nhãn gốc để mô hình học sự đa dạng
            label = demo["label"] 
            
            demo_text += f'\n[Ví dụ {i}]\nNội dung: "{demo["text"].strip()}..."\nKết luận (JSON): {{"label": "{label}", "explanation": "[Kết luận] vì [Lý do/Bằng chứng]"}}\n'
    else:
        demo_text = "\n(Không có ví dụ)\n"

    user_prompt = FINAL_CLASSIFICATION_USER_PROMPT_TEMPLATE.format(
        rag_text=rag_text,
        wiki_text=wiki_text,
        demo_text=demo_text.strip(),
        text_input=text_input.strip()
    )
    return FINAL_CLASSIFICATION_SYSTEM_PROMPT, user_prompt
