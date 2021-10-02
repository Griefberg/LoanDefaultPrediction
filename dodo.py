def task_fmt():
    """run all formatters"""
    return {"actions": ["isort .", "black -q ."]}
