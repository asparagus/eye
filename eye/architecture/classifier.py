# from jigsaw.piece import Piece

# from eye.architecture.names import (
#     STATE
# )


# class Classifier(Piece):

#     def __init__(self, num_filters: int, num_classes: int):
#         super().__init__(
#             piece_type="module",
#             name="classifier",
#         )
#         self.num_filters = num_filters
#         self.num_classes = num_classes

#     def inputs(self) -> tuple[str, ...]:
#         return (IMAGE_INPUT, FOCUS_POINT)

#     def outputs(self) -> tuple[str, ...]:
#         return (STATE, FOCUS_HISTORY, MOTOR_LOSS)
