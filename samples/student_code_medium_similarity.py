"""
Student Code: Medium Similarity Example
This student uses different naming and structure but achieves similar functionality.
"""


class Calculator:
    """Calculator class with basic operations."""

    @staticmethod
    def sum_numbers(num1, num2):
        """Add two numbers."""
        result = num1 + num2
        return result

    @staticmethod
    def difference(num1, num2):
        """Find difference between numbers."""
        result = num1 - num2
        return result

    @staticmethod
    def product(num1, num2):
        """Calculate product."""
        result = num1 * num2
        return result

    @staticmethod
    def quotient(num1, num2):
        """Calculate quotient."""
        if num2 == 0:
            print("Error: Division by zero")
            return None
        result = num1 / num2
        return result

    def perform_operation(self, symbol, num1, num2):
        """Perform calculation based on symbol."""
        if symbol == '+':
            return self.sum_numbers(num1, num2)
        elif symbol == '-':
            return self.difference(num1, num2)
        elif symbol == '*':
            return self.product(num1, num2)
        elif symbol == '/':
            return self.quotient(num1, num2)
        else:
            print(f"Unknown symbol: {symbol}")
            return None


def run_tests():
    """Run calculator tests."""
    calc = Calculator()

    print("My Calculator")
    print("=" * 40)

    test1 = calc.perform_operation('+', 5, 3)
    print(f"5 + 3 = {test1}")

    test2 = calc.perform_operation('-', 10, 4)
    print(f"10 - 4 = {test2}")

    test3 = calc.perform_operation('*', 6, 7)
    print(f"6 * 7 = {test3}")

    test4 = calc.perform_operation('/', 20, 4)
    print(f"20 / 4 = {test4}")


if __name__ == "__main__":
    run_tests()
