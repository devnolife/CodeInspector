"""
Student Code: High Similarity Example
This student's code is very similar to the reference implementation.
"""


def add(x, y):
    """Add two numbers together."""
    return x + y


def subtract(x, y):
    """Subtract y from x."""
    return x - y


def multiply(x, y):
    """Multiply two numbers."""
    return x * y


def divide(x, y):
    """Divide x by y."""
    if y == 0:
        raise ValueError("Division by zero is not allowed")
    return x / y


def calculate(op, x, y):
    """
    Execute calculation based on operation.

    Args:
        op: Operation to perform
        x: First number
        y: Second number

    Returns:
        Result of calculation
    """
    if op == '+':
        return add(x, y)
    elif op == '-':
        return subtract(x, y)
    elif op == '*':
        return multiply(x, y)
    elif op == '/':
        return divide(x, y)
    else:
        raise ValueError(f"Invalid operation: {op}")


def main():
    """Main program to test calculator."""
    print("Calculator Program")
    print("-" * 40)

    print(f"5 + 3 = {calculate('+', 5, 3)}")
    print(f"10 - 4 = {calculate('-', 10, 4)}")
    print(f"6 * 7 = {calculate('*', 6, 7)}")
    print(f"20 / 4 = {calculate('/', 20, 4)}")


if __name__ == "__main__":
    main()
