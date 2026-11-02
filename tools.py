import ast
import math
import operator
import re
from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parent / "docs"

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def calculator(expression: str) -> str:
    """Calculate arithmetic safely, without eval()."""

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            result = node.value

        elif isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            left = evaluate(node.left)
            right = evaluate(node.right)

            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("слишком большая степень")

            result = OPERATORS[type(node.op)](left, right)

        elif isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
            result = OPERATORS[type(node.op)](evaluate(node.operand))

        else:
            raise ValueError(
                "допустимы только числа и операторы + - * / % **"
            )

        if (
            type(result) not in (int, float)
            or abs(result) > 1e100
            or not math.isfinite(result)
        ):
            raise ValueError("результат вне допустимых пределов")

        return result

    try:
        expression = expression.strip().strip("`'\"").replace("^", "**")

        if len(expression) > 500:
            raise ValueError("выражение слишком длинное")

        tree = ast.parse(expression, mode="eval")

        if sum(1 for _ in ast.walk(tree)) > 100:
            raise ValueError("выражение слишком сложное")

        result = evaluate(tree.body)

        if isinstance(result, float):
            return str(round(result, 6))

        return str(result)

    except ZeroDivisionError:
        return "Ошибка: деление на ноль"
    except Exception as error:
        return f"Ошибка: {error}"


def _tokenize(text: str) -> set[str]:
    """Extract unique lowercase words longer than two characters."""
    return {
        word
        for word in re.findall(r"\w+", text.lower())
        if len(word) > 2
    }


def text_search(query: str, top_k: int = 3) -> str:
    """Return paragraphs ranked by the number of shared query words."""
    if type(top_k) is not int or top_k < 1:
        return "Ошибка: top_k должен быть положительным целым числом"

    query_words = _tokenize(query)

    if not query_words:
        return "Ничего не найдено. Укажите более точные ключевые слова."

    try:
        files = sorted(DOCS_DIR.glob("*.txt"))

        if not files:
            return "Ошибка: в папке docs нет файлов .txt"

        hits = []

        for path in files:
            text = path.read_text(encoding="utf-8")
            paragraphs = re.split(r"\n\s*\n", text.strip())

            for index, paragraph in enumerate(paragraphs, start=1):
                paragraph_words = _tokenize(paragraph)
                score = len(query_words & paragraph_words)

                if score > 0:
                    hits.append((score, path.name, index, paragraph))

        if not hits:
            return (
                "Ничего не найдено. Переформулируйте запрос "
                "или попробуйте другие ключевые слова."
            )

        # Highest score first; deterministic ordering for equal scores.
        hits.sort(key=lambda hit: (-hit[0], hit[1], hit[2]))

        results = []

        for score, filename, index, paragraph in hits[:top_k]:
            snippet = paragraph.replace("\n", " ")

            if len(snippet) > 400:
                snippet = snippet[:397] + "..."

            results.append(
                f"[{filename}, абзац {index}] {snippet}"
            )

        return "\n".join(results)

    except (OSError, UnicodeError) as error:
        return f"Ошибка чтения документов: {error}"