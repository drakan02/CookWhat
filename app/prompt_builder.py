from typing import Any, Dict, List, Optional
import re


def _sanitize_prompt_text(text: Optional[str]) -> str:
    """Sanitize user-provided text before injecting into prompt."""
    if text is None:
        return ""
    safe = str(text).strip()
    safe = safe.replace("\r", " ")
    safe = safe.replace("\n", " ")
    safe = re.sub(r"\s+", " ", safe)
    return safe[:1000]


def _format_chat_history(chat_history: Optional[List[Dict[str, str]]]) -> str:
    """Format chat history for injection into the prompt."""
    if not chat_history:
        return ""
    lines = []
    for msg in chat_history:
        role = "Người dùng" if msg.get("role") == "user" else "CookWhat AI"
        lines.append(f"{role}: {msg.get('content', '').strip()}")
    return "\n".join(lines)


def build_prompt(
    user_ingredients: List[str],
    vector_results: List[Dict[str, Any]],
    user_request: Optional[str] = None,
    nutrition_context: Optional[Dict[str, Any]] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Build the prompt for the LLM based on ingredients, recipe results and chat history."""
    context_text = ""
    nutrition_context = nutrition_context or {}
    user_request = _sanitize_prompt_text(user_request)
    history_text = _format_chat_history(chat_history)

    for i, recipe in enumerate(vector_results, 1):
        context_text += f"""
========================
CÔNG THỨC {i}

Tên món: {recipe.get('title')}
URL: {recipe.get('url')}

Thông tin công thức:
{recipe.get('document')}
========================
"""

    history_section = f"""
Lịch sử trò chuyện gần đây:
{history_text}
""" if history_text else ""

    prompt = f"""Bạn là CookWhat AI — trợ lý nấu ăn thông minh bằng tiếng Việt. Hãy trả lời **bằng Markdown**, tự nhiên và thân thiện như một người bạn đang trò chuyện.
{history_section}
Tin nhắn mới nhất của người dùng:
"{user_request or ', '.join(user_ingredients)}"

Nguyên liệu người dùng cung cấp: {', '.join(user_ingredients)}

Hệ thống tìm được {len(vector_results)} công thức từ Cookpad để tham khảo:

{context_text}

Dữ liệu dinh dưỡng nội bộ để ước lượng:
{_format_nutrition_context(nutrition_context)}

---

**Hướng dẫn trả lời:**

- Đọc kỹ lịch sử trò chuyện và tin nhắn mới nhất để hiểu người dùng đang hỏi gì.
- Trả lời **đúng trọng tâm** câu hỏi — không nhất thiết phải liệt kê tất cả {len(vector_results)} món. Ví dụ:
  - Nếu người dùng hỏi "món nào nhanh nhất?" → chỉ trả lời món nhanh nhất.
  - Nếu người dùng hỏi "gợi ý cho mình vài món" → liệt kê một vài món phù hợp.
  - Nếu người dùng hỏi về một món cụ thể → tập trung vào món đó.
- Chỉ sử dụng các công thức trong dữ liệu được cung cấp; không tự bịa công thức mới từ Cookpad.
- Không nhắc tới "vector database", điểm số, score, similarity hay bất kỳ thuật ngữ kỹ thuật nào.
- Không trả JSON hay dữ liệu thô.
- Dùng Markdown: tiêu đề, bullet points, in đậm để dễ đọc.

**Khi giới thiệu một món, bao gồm (nếu phù hợp):**
- Giới thiệu ngắn vì sao món đó phù hợp với nguyên liệu hiện tại
- Nguyên liệu người dùng đang có
- Còn thiếu nguyên liệu gì không — nếu có thì ghi rõ và gợi ý mua thêm
- Ước lượng calo mỗi phần ăn theo dạng khoảng, ví dụ "Ước lượng calo: khoảng 450–600 kcal/phần"; nếu dữ liệu thiếu định lượng thì vẫn ước lượng hợp lý và ghi ngắn "chỉ là ước lượng"
- Thời gian nấu
- Tóm tắt cách làm dễ hiểu
- Link công thức Cookpad (URL thuần, ví dụ: https://cookpad.com/...)

**Quy tắc dinh dưỡng:**
- Với câu trả lời gợi ý nhiều món: mỗi món chỉ cần "Ước lượng calo" — không liệt kê protein, fat, carb, sodium, fiber hoặc bảng dinh dưỡng chi tiết.
- Nếu người dùng hỏi sâu về dinh dưỡng của một món cụ thể, hãy trả thêm các nutrient quan trọng: calo, protein, chất béo, carb, chất xơ, đường, sodium nếu có thể ước lượng.
- Ưu tiên dùng dữ liệu dinh dưỡng nội bộ; nếu không có dữ liệu phù hợp thì tự ước lượng hợp lý.
- Luôn chỉ ghi là "ước lượng" trong câu trả lời — không nhắc nguồn dữ liệu hoặc "AI ước lượng".

**Nếu có ràng buộc đặc biệt** (không có bếp, không có lửa, ăn lạnh, ăn sống, ăn chay...):
- Ưu tiên đánh giá món nào có thể làm không cần gia nhiệt.
- Cảnh báo rõ món nào trong dữ liệu không phù hợp vì cần nấu/chiên/áp chảo.
- Không gợi ý món dùng thịt bò sống/trứng sống nếu không an toàn; nếu có nhắc thì phải cảnh báo rủi ro an toàn thực phẩm.
- Nếu tất cả công thức trong dữ liệu không phù hợp, nói rõ không có công thức phù hợp; sau đó có thể đưa 1–2 ý tưởng an toàn ngoài dữ liệu nhưng phải ghi rõ đó chỉ là gợi ý chung, không phải công thức từ Cookpad.

**Cuối câu trả lời** (nếu phù hợp):
- Đưa ra lời khuyên nên thử món nào trước.
- Hoặc hỏi người dùng có muốn món healthy / nhanh / ít dầu mỡ hơn không.
"""
    return prompt


def _format_nutrition_context(nutrition_context: Dict[str, Any]) -> str:
    if not nutrition_context:
        return "Không có dữ liệu phù hợp. Nếu cần calo, hãy tự ước lượng hợp lý."

    lines: List[str] = []
    for query, item in nutrition_context.items():
        parts = [
            f"- {query}: dữ liệu gần nhất '{item.get('matched_name')}'",
            f"kcal/100g={item.get('kcal_per_100g')}",
        ]
        if item.get("protein_g_per_100g") is not None:
            parts.append(f"protein/100g={item.get('protein_g_per_100g')}g")
        if item.get("fat_g_per_100g") is not None:
            parts.append(f"fat/100g={item.get('fat_g_per_100g')}g")
        if item.get("carb_g_per_100g") is not None:
            parts.append(f"carb/100g={item.get('carb_g_per_100g')}g")
        lines.append("; ".join(parts))

    return "\n".join(lines)
