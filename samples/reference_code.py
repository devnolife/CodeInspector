"""
Reference Code: Simple Calculator
This is the expected implementation for the calculator assignment.
"""


def add(a, b):
    """Add two numbers."""
    return a + b


def subtract(a, b):
    """Subtract b from a."""
    return a - b


def multiply(a, b):
    """Multiply two numbers."""
    return a * b


def divide(a, b):
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def calculate(operation, a, b):
    """
    Perform a calculation based on the operation.

    Args:
        operation: The operation to perform (+, -, *, /)
        a: First operand
        b: Second operand

    Returns:
        Result of the calculation
    """
    if operation == '+':
        return add(a, b)
    elif operation == '-':
        return subtract(a, b)
    elif operation == '*':
        return multiply(a, b)
    elif operation == '/':
        return divide(a, b)
    else:
        raise ValueError(f"Unknown operation: {operation}")


def main():
    """Main function to demonstrate calculator usage."""
    print("Simple Calculator")
    print("-" * 40)

    # Test cases
    print(f"5 + 3 = {calculate('+', 5, 3)}")
    print(f"10 - 4 = {calculate('-', 10, 4)}")
    print(f"6 * 7 = {calculate('*', 6, 7)}")
    print(f"20 / 4 = {calculate('/', 20, 4)}")


if __name__ == "__main__":
    main()
