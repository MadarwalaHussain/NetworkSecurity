import sys
from networksecurity.logging import logger

class NetworkSecurityException(Exception):
    """
    Base class for all exceptions in the NetworkSecurity package.
    """
    def __init__(self, error_message, error_details:sys):
        self.error_message = error_message
        _,_, exc_tb = error_details.exc_info()

        self.lineno = exc_tb.tb_lineno
        self.filename = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return f"{self.error_message} occurred in file {self.filename} at line number {self.lineno}"
    

# if __name__ == "__main__":
#     try:
#         logger.logging.info("This is a test log message.")
#         # Simulating an exception for demonstration purposes
#         a = 1/0

#     except Exception as e:
#         raise NetworkSecurityException(
#             error_message=e,
#             error_details=sys)
#         # Output: This is a test exception occurred in file <filename> at line number <lineno>