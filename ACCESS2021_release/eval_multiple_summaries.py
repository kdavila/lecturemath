
import sys

from AccessMath.preprocessing.user_interface.console_ui_process import ConsoleUIProcess
from AccessMath.evaluation.summary_evaluator import SummaryEvaluator


def main():
    # usage check
    if not ConsoleUIProcess.usage_with_config_check(sys.argv):
        print("\t-b [baseline]\t: Process Summaries from specified baseline")
        return

    process = ConsoleUIProcess.FromConfigPath(sys.argv[1], sys.argv[2:], None, None)
    if not process.initialize():
       return

    evaluator = SummaryEvaluator()
    process.start_input_processing(evaluator.process_summary)

    evaluator.print_totals(True)

    print("Finished")


if __name__ == "__main__":
    main()
