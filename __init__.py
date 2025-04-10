# insightagent/__init__.py

from .tools.tools import (
    upload_file,
    clean_data,
    detect_column_types,
    generate_summary,
    suggest_charts,
    build_dashboard,
    export_report,
    run_prediction,
    save_to_memory,
    retrieve_from_memory,
    strategy_recommender,
    detect_kpi_candidates,
    data_storytelling_planner,
    dashboard_autotuner,
    insight_validator
)

TOOLS = [
    upload_file,
    clean_data,
    detect_column_types,
    generate_summary,
    suggest_charts,
    build_dashboard,
    export_report,
    run_prediction,
    save_to_memory,
    retrieve_from_memory,
    strategy_recommender,
    detect_kpi_candidates,
    data_storytelling_planner,
    dashboard_autotuner,
    insight_validator
]
