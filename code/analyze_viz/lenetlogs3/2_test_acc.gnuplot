

reset
set terminal png
set output "test_acc.png"
set style data lines
set key right

###### Fields in the data file your_log_name.log.train are
###### Iters Seconds TrainingLoss LearningRate

# Training loss vs. training iterations
set title "Test accuracy vs. training iterations"
set xlabel "Test accuracy"
set ylabel "training iterations"
# plot "alexlog.log.train" using 1:3 title "lenet"

# Training loss vs. training time
# plot "alexlog.log.train" using 2:3 title "alexnet"

# Learning rate vs. training iterations;
# plot "alexlog.log.train" using 1:4 title "alexnet"

# Learning rate vs. training time;
# plot "alexlog.log.train" using 2:4 title "alexnet"


###### Fields in the data file your_log_name.log.test are
###### Iters Seconds TestAccuracy TestLoss

# Test loss vs. training iterations
# plot "alexlog.log.test" using 1:4 title "alexnet"

# Test accuracy vs. training iterations
plot "log-lenet.log.test" using 1:3 title "lenet"

# Test loss vs. training time
# plot "alexlog.log.test" using 2:4 title "alexnet"

# Test accuracy vs. training time
# plot "alexlog.log.test" using 2:3 title "alexnet"


