"""
Student Code: Low Similarity Example
This student implemented a completely different approach.
"""

import operator


def evaluate_expression(expression):
    """
    Evaluate a mathematical expression string.

    Args:
        expression: String like "5 + 3" or "10 / 2"

    Returns:
        Result of evaluation
    """
    # Split the expression
    parts = expression.strip().split()

    if len(parts) != 3:
        return "Invalid expression format"

    num1 = float(parts[0])
    op_symbol = parts[1]
    num2 = float(parts[2])

    # Dictionary mapping symbols to operations
    operations = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv
    }

    if op_symbol not in operations:
        return f"Unknown operator: {op_symbol}"

    try:
        result = operations[op_symbol](num1, num2)
        return result
    except ZeroDivisionError:
        return "Error: Division by zero"


def interactive_calculator():
    """Run calculator in interactive mode."""
    print("Expression Calculator")
    print("Enter expressions like: 5 + 3")
    print("-" * 40)

    test_expressions = [
        "5 + 3",
        "10 - 4",
        "6 * 7",
        "20 / 4"
    ]

    for expr in test_expressions:
        result = evaluate_expression(expr)
        print(f"{expr} = {result}")


if __name__ == "__main__":
    interactive_calculator()
