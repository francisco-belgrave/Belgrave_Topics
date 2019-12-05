# Examples and Exercises from Think Stats, 2nd Edition

# http://thinkstats2.com

# Copyright 2016 Allen B. Downey

# MIT License: https://opensource.org/licenses/MIT


from __future__ import print_function, division

from Resources.Think_Stats.Thinkstats2 import nsfg

## Examples from Chapter 1

# Read NSFG data into a Pandas DataFrame.

preg = nsfg.ReadFemPreg()
preg.head()

# Print the column names.

preg.columns

# Select a single column name.

preg.columns[1]

# Select a column and check what type it is.

pregordr = preg['pregordr']
type(pregordr)

# Print a column.

pregordr

# Select a single element from a column.

pregordr[0]

# Select a slice from a column.

pregordr[2:5]

# Select a column using dot notation.

pregordr = preg.pregordr

# Count the number of times each value occurs.

preg.outcome.value_counts().sort_index()

# Check the values of another variable.

preg.birthwgt_lb.value_counts().sort_index()

# Make a dictionary that maps from each respondent's `caseid` to a list of indices into the pregnancy `DataFrame`.  Use it to select the pregnancy outcomes for a single respondent.

caseid = 10229
preg_map = nsfg.MakePregMap(preg)
indices = preg_map[caseid]
preg.outcome[indices].values

## Exercises

# Select the `birthord` column, print the value counts, and compare to results published in the [codebook](http://www.icpsr.umich.edu/nsfg6/Controller?displayPage=labelDetails&fileCode=PREG&section=A&subSec=8016&srtLabel=611933)

# Solution

preg.birthord.value_counts().sort_index()

# We can also use `isnull` to count the number of nans.

preg.birthord.isnull().sum()

# Select the `prglngth` column, print the value counts, and compare to results published in the [codebook](http://www.icpsr.umich.edu/nsfg6/Controller?displayPage=labelDetails&fileCode=PREG&section=A&subSec=8016&srtLabel=611931)

# Solution

preg.prglngth.value_counts().sort_index()

# To compute the mean of a column, you can invoke the `mean` method on a Series.  For example, here is the mean birthweight in pounds:

preg.totalwgt_lb.mean()

# Create a new column named <tt>totalwgt_kg</tt> that contains birth weight in kilograms.  Compute its mean.  Remember that when you create a new column, you have to use dictionary syntax, not dot notation.

# Solution

preg['totalwgt_kg'] = preg.totalwgt_lb / 2.2
preg.totalwgt_kg.mean()

# `nsfg.py` also provides `ReadFemResp`, which reads the female respondents file and returns a `DataFrame`:

resp = nsfg.ReadFemResp()

# `DataFrame` provides a method `head` that displays the first five rows:

resp.head()

# Select the `age_r` column from `resp` and print the value counts.  How old are the youngest and oldest respondents?

# Solution

resp.age_r.value_counts().sort_index()

# We can use the `caseid` to match up rows from `resp` and `preg`.  For example, we can select the row from `resp` for `caseid` 2298 like this:

resp[resp.caseid==2298]

# And we can get the corresponding rows from `preg` like this:

preg[preg.caseid==2298]

# How old is the respondent with `caseid` 1?

# Solution

resp[resp.caseid==1].age_r

# What are the pregnancy lengths for the respondent with `caseid` 2298?

# Solution

preg[preg.caseid==2298].prglngth

# What was the birthweight of the first baby born to the respondent with `caseid` 5012?

# Solution

preg[preg.caseid==5012].birthwgt_lb

