"""
Run FairTest Testing Investigation on Adult Dataset

Usage: python adult_fairness_test.py
"""

import sys
import os
from data_generator.main import get_real_data
from methods.group.fairtest import Testing, train, test, report, DataSource


def main(argv=sys.argv):
    if len(argv) != 1:
        usage(argv)

    # Load the Adult dataset
    discrimination_data, data_schema = get_real_data('adult', use_cache=True)
    OUTPUT_DIR = "."

    # Initializing parameters for experiment
    # From the get_real_data function, 'race' and 'sex' are the protected columns
    # and 'income' is the outcome.
    SENS = discrimination_data.protected_attributes
    TARGET = discrimination_data.outcome_column
    EXPL = []

    data_source = DataSource(discrimination_data.training_dataframe)

    # Instantiate the experiment
    inv = Testing(data_source, SENS, TARGET, EXPL, random_state=0)

    # Train the classifier
    train([inv])

    # Evaluate on the testing set
    test([inv])

    # Create the report
    report([inv], "adult_testing", OUTPUT_DIR)


def usage(argv):
    print("Usage:%s" % argv[0])
    exit(-1)

if __name__ == '__main__':
    sys.exit(main())
