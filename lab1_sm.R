##Exercise 1
##For x = 3 and y = 15, compute the sum, difference, product and quotient of x and y
x <- 3
y <- 15
x+y
x-y
x*y
x/y

##Exercise 2
##For x = 2 and y = 3, Test x^5 is greater than y^4 and return a logical value
x <- 2; y <- 3
z = x^5 > y^4
z

##Exercise 3
##Create a vector of the values 22, 62, 148, 43 and 129. Determine the sum of the vector. Divide each value in the vector by the sum to determine relate frequency.
x<- c(22,62,148,43,129)
sum(x)
x/(sum(x))

##Exercise 4
##Create a vector of the nucleotides A, T, C and G. Sort the vector.
x<- c("A","T","C","G")
sort(x)