# Possible combinations:
# 1. Test accuracy (test score 0) vs. training iterations / time;
# 2. Test loss (test score 1) time;
# 3. Training loss vs. training iterations / time;
# 4. Learning rate vs. training iterations / time;
# A rarer one: Training time vs. iterations.

# What is the difference between plotting against iterations and time?
# If the overhead in one iteration is too high, one algorithm might appear
# to be faster in terms of progress per iteration and slower when measured
# against time. And the reverse case is not entirely impossible. Thus, some
# papers chose to only publish the more favorable type. It is your freedom
# to decide what to plot.

reset
set terminal png
set output "train_loss.png"
set style data lines
set key right

###### Fields in the data file your_log_name.log.train are
###### Iters Seconds TrainingLoss LearningRate

# Training loss vs. training iterations
set title "Training loss vs. training iterations"
set xlabel "Training iterations"
set ylabel "Training loss"
plot "log-lenet.log.train" using 1:3 title "lenet-train" ,\
    "log-lenet.log.test" using 1:4 title "lenet-val"

# Training loss vs. training time
# plot "alexlog.log.train" using 2:3 title "lenet"

# Learning rate vs. training iterations;
# plot "alexlog.log.train" using 1:4 title "lenet"

# Learning rate vs. training time;
# plot "alexlog.log.train" using 2:4 title "lenet"


###### Fields in the data file your_log_name.log.test are
###### Iters Seconds TestAccuracy TestLoss

# Test loss vs. training iterations
# plot "alexlog.log.test" using 1:4 title "lenet"

# Test accuracy vs. training iterations
# plot "alexlog.log.test" using 1:3 title "lenet"

# Test loss vs. training time
# plot "alexlog.log.test" using 2:4 title "lenet"

# Test accuracy vs. training time
# plot "alexlog.log.test" using 2:3 title "lenet"
