import pandas as pd

from src.utils.dataset import analyze_column_types


def construct_prompt_name_description(
    X,
    target_name,
    dataset_info,
    history,
    context_template,
    instructions,
    include_ic_samples,
    include_history,
    include_feature_description,
    use_description_in_history=True,
    n_ic_samples=2,
):
    """Handle construct prompt name description."""
    prompt_context = context_template.format(
        target_name=target_name, dataset_info=dataset_info
    )

    prompt_feature_description = serialize_features(X)

    prompt_in_context = serialize_in_context(X, n_ic_samples)

    prompt_history = serialize_history(
        history,
        include_history=include_history,
        use_description=use_description_in_history,
    )

    prompt = prompt_context
    if include_feature_description:
        prompt += "\n\n" + prompt_feature_description
    if include_ic_samples:
        prompt += "\n\n" + prompt_in_context

    prompt += "\n\n" + prompt_history

    prompt += "\n\n" + instructions

    return prompt


def construct_prompt_reflection(
    X,
    list_features,
    target_name,
    dataset_info,
    history,
    context_template,
    instructions,
    include_ic_samples,
    include_history,
    include_feature_description,
    sort_by_score=True,
    n_ic_samples=2,
    include_code=False,
    include_rationale=False,
    selector=None,
    n_parents_per_prompt=1,
):
    """Handle construct prompt reflection."""
    list_parents = selector.select(list_features, n_parents_per_prompt)

    prompt_context = context_template.format(
        target_name=target_name, dataset_info=dataset_info
    )  # "This is a dataset about RNA polymerase, your task is to ..."

    prompt_feature_description = serialize_features(X)

    prompt_in_context = serialize_in_context(X, n_ic_samples)

    prompt_history = serialize_history(history, include_history)

    prompt_population = serialize_list_features(
        list_parents,
        sort_by_score=sort_by_score,
        include_code=include_code,
        include_rationale=include_rationale,
    )

    prompt = prompt_context
    if include_feature_description:
        prompt += "\n\n" + prompt_feature_description
    if include_ic_samples:
        prompt += "\n\n" + prompt_in_context

    prompt += "\n\n" + instructions

    prompt += "\n\n" + prompt_population

    prompt += "\n\n" + prompt_history

    return prompt


def serialize_list_features(
    list_features,
    sort_by_score=True,
    include_code=False,
    include_rationale=False,
):
    # Initialize prompt string
    """Handle serialize list features."""
    prompt = "<Beginning of the population of features> \n"
    prompt += "Here is the list of features along with their score: \n"
    if sort_by_score:
        list_features = sorted(list_features, key=lambda x: x.score)
    # Iterate through selected samples
    for i, feature in enumerate(list_features):
        # Start new sample
        prompt += f"Feature {i + 1} \n"
        prompt += f"Score: {feature.score:.4f} \n Feature name: {feature.name} \n Feature description: {feature.description}"
        if include_rationale:
            prompt += f"\n  Rationale: {feature.rationale}"
        if include_code:
            prompt += f"\n Feature code: {feature.string}"

    

        # Add a new line
        prompt += "\n"

        # Remove trailing comma and add newline
        prompt = prompt.rstrip(",")

    prompt += "<End of the population of features>"

    return prompt.rstrip()


def serialize_in_context(X, n_samples):
    """
    Constructs a prompt with randomly selected samples from a DataFrame.
    """
    # Randomly select n_samples rows from X
    selected_samples = X.sample(n=min(n_samples, len(X)))

    # Initialize prompt string
    prompt = "To help you, I am also providing some randomly sampled examples from the dataset: \n"

    # Iterate through selected samples
    for i, (_, row) in enumerate(selected_samples.iterrows(), 1):
        # Start new sample
        prompt += f"Sample {i}:"

        # Add feature-value pairs
        for feature_name, value in row.items():
            prompt += f" {feature_name}: {value},"

        # Remove trailing comma and add newline
        prompt = prompt.rstrip(",") + "\n"

    # Add the size of the datasets
    prompt += f"\n The dataset has {len(X)} samples."

    return prompt.rstrip()


def serialize_features(X):
    """
    Creates a description of each feature in the DataFrame including type and statistics.
    """
    # Get column types using the provided function
    column_types = analyze_column_types(X)

    # Initialize the description string
    descriptions = []

    for column in X.columns:
        col_type = column_types[column]

        # Start with basic feature name and type
        description = f"{column}: {col_type}"

        # Add additional information based on column type
        if col_type == "continuous" or col_type == "discrete":
            min_val = X[column].min()
            max_val = X[column].max()
            description += f" (range: {min_val} to {max_val})"

        elif col_type == "categorical" or col_type == "boolean":
            unique_values = sorted(X[column].unique())
            # Convert to strings and handle potential NaN values
            unique_values = [str(val) for val in unique_values if pd.notna(val)]
            description += f" (values: {', '.join(unique_values)})"

        elif col_type == "datetime":
            min_date = X[column].min()
            max_date = X[column].max()
            description += f" (range: {min_date} to {max_date})"

        elif col_type == "text":
            avg_length = X[column].str.len().mean()
            description += f" (average length: {avg_length:.1f} characters)"

        descriptions.append(description)

    # Join all descriptions with newlines
    return "\n".join(descriptions)


def serialize_history(history, include_history, use_description=True):
    """
    Serializes the decision tree path history into a readable format.
    """
    if not history or not include_history:
        return "<Beginning Splitting conditions from root to current node> \n This is the root node (no conditions applied). <End of Splitting conditions from root to current node>"

    # Start with the header
    result = "<Beginning Splitting conditions from root to current node> \n"

    # Add each condition
    for feature, operator in history:
        # Format the condition
        condition = f"{feature.name} {operator} than {feature.threshold:.3f}"
        if use_description:
            condition += f" ({feature.description})"
        result += condition + "\n"

    # Add the end of the history
    result += "<End of Splitting conditions from root to current node>"

    return result.rstrip()


def serialize_proposed_feature(feature):
    """
    Serializes a proposed feature into a formatted string.
    """
    template = (
        "The feature you should generate has the following characteristics:\n"
        "Feature name: {name}\n"
        "Feature description: {description}"
    )

    return template.format(name=feature.name, description=feature.description)


def construct_prompt_code_generation(X, context, proposed_feature, instructions):
    """Handle construct prompt code generation."""
    prompt_proposed_feature = serialize_proposed_feature(proposed_feature)
    prompt_features = serialize_features(X)

    prompt = (
        context
        + "\n\n You should build this feature using the following original features:\n"
        + prompt_features
        + "\n\n"
        + prompt_proposed_feature
        + "\n\n"
        + instructions
    )
    return prompt

