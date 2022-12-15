import torch.nn as nn


class LinearClassifier(nn.Module):
    """
    The linear classifier module.
    Script builds it based on passed parameters.

    Arguments:
        input_dim (int) - Size of input.
        hidden_dims (list[int]) - Sizes of hidden layers.
        output_dim (int) - Size of output.
        activation_function_name (str) - Name of a activation function, which will be used in classifier.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        activation_function_name: str,
    ):
        super().__init__()

        self.classifier = self.build_classifier(
            input_dim, hidden_dims, output_dim, activation_function_name
        )

    def build_classifier(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        activation_function_name: str,
    ):
        """
        The function builds list, which contains linear layers and returns nn.Sequential.

        Arguments:
            input_dim (int) - Size of input.
            hidden_dims (list[int]) - Sizes of hidden layers.
            output_dim (int) - Size of output.
            activation_function_name (str) - Name of a activation function, which will be used in classifier.

        Returns:
            model (nn.Sequential) - The classifier.
        """

        layers = []

        activation_function = getattr(nn, activation_function_name)()

        # If user doesn't pass hidden layers, create only input and output layer.
        if len(hidden_dims) == 0:
            layers.append(nn.Linear(in_features=input_dim, out_features=output_dim))
        else:
            # Add first layer and activation function
            layers.append(nn.Linear(in_features=input_dim, out_features=hidden_dims[0]))
            layers.append(activation_function)

            # Add hidden layers in appropriate sizes
            for i in range(1, len(hidden_dims)):
                layers.append(
                    nn.Linear(
                        in_features=hidden_dims[i - 1], out_features=hidden_dims[i]
                    )
                )
                layers.append(activation_function)

            # Add last layer
            layers.append(
                nn.Linear(in_features=hidden_dims[-1], out_features=output_dim)
            )

        model = nn.Sequential(*layers)
        return model
