using Plots
using ProximalOperators
using LinearAlgebra
using Statistics
using Random

function svm(x, y, lambda, sigma)
	iter = 10000
	println("Computing Kernel")
	K = kernel(x, sigma)
	Y = Diagonal(y)
	println("Computing Q")
	Q = (1/lambda) * (Y * K * Y)
	step_size = 1/opnorm(Q)
	h = HingeLoss(ones(length(x)),1/length(x))
	h_conj = Conjugate(h)
	v = zeros(length(x), iter+1)
	println("Starting prox iterations")
	for i=2:iter+1
		grad = Q * v[:, i-1]
		v[:, i], _ = prox(h_conj, (v[:,i-1] - step_size*grad), step_size)
	end
	return v
end

function pred_model(x_hat, x, y, v, lambda, sigma)
	y_k = zeros(length(x),1)
	for i=1:length(x)
		x_diff = x[i]-x_hat
		x_diff_norm = x_diff'*x_diff
		c = -(1/(2*sigma^2))
		y_k[i] = exp(c*x_diff_norm)*y[i]
	end
	m_out = (-1/lambda)*v'*y_k
	if m_out[1] < 0.0
		return -1.0
	else
		return 1.0
	end

end

function kernel(x, sigma)
	d = ones(length(x))
	D = Diagonal(d)
	K = zeros(length(x),length(x)) + D
	for i=1:length(x)-1
		for j=i+1:length(x)
			x_diff = x[i]-x[j]
			x_diff_norm = x_diff'*x_diff
			c = -(1/(2*sigma^2))
			k = exp(c*x_diff_norm)
			K[i,j] = k
			K[j,i] = k
		end
	end
	return K
end

function task_7(v_data,v_y)
	lambda = [0.1, 0.01, 0.001, 0.0001]
	sigma = [1, 0.5, 0.25]
	x,y = svm_train()
	test_errors = zeros(length(lambda),length(sigma))
	train_errors = zeros(length(lambda),length(sigma))
	for l=1:length(lambda)
		for s=1:length(sigma)
			println("lambda=$(lambda[l]) \n sigma=$(sigma[s])")
			v = svm(x, y, lambda[l], sigma[s])
			v = v[:,end]
			#print(sqrt.(sum(abs2, v, dims=1)))
			x_test_labels = zeros(length(v_data))
			for i=1:length(v_data)
				x_test_labels[i] = pred_model(v_data[i], x, y, v, lambda[l], sigma[s])
			end
			x_train_labels = zeros(length(x))
			for i=1:length(x)
				x_train_labels[i] = pred_model(x[i], x, y, v, lambda[l], sigma[s])
			end
			#v_y, true labels, x_hat_class, predicted labels
			#Training data error.
			test_error = mean(x_test_labels .!= v_y)
			train_error = mean(x_train_labels .!= y)
			test_errors[l,s] = test_error
			train_errors[l,s] = train_error
			println("test_error: $test_error")
			println("train_error: $train_error")
			println("\n")
		end
	end
	return test_errors, train_errors, lambda, sigma
end

function plotter(test_errors, train_errors, lambdas, sigmas)
	#test_errors, train_errors, lambdas, sigmas = task_7()
	for s=1:length(sigmas)
		plot(lambdas,test_errors[:,s],
		 	xlabel="lambda",
			ylabel="error",
			title="Sigma=$(sigmas[s])",
			label="test",
			xscale=:log10,
			ylims = (0,0.32),
			linecolor=:blue)
		plot!(lambdas, test_errors[:,s], label="test",seriestype=:scatter,color=:blue)
		plot!(lambdas, train_errors[:,s],
			label="train", color=:green, seriestype=:scatter)
		plot!(lambdas, train_errors[:,s],label="train", color=:green)
		savefig("svm_error_lambdas_$(sigmas[s])")
	end
end

function task_8(v_data, v_y)
	lambda = 0.001
	sigma = 0.5
	x,y = svm_train()
	v = svm(x, y, lambda, sigma)
	v = v[:,end]
	#print(sqrt.(sum(abs2, v, dims=1)))
	x_test_labels = zeros(length(v_data))
	for i=1:length(v_data)
		x_test_labels[i] = pred_model(v_data[i], x, y, v, lambda, sigma)
	end
	x_train_labels = zeros(length(x))
	for i=1:length(x)
		x_train_labels[i] = pred_model(x[i], x, y, v, lambda, sigma)
	end
	#v_y, true labels, x_hat_class, predicted labels
	#Training data error.
	test_error = mean(x_test_labels .!= v_y)
	train_error = mean(x_train_labels .!= y)
	println("test_error: $test_error")
	println("train_error: $train_error")
	println("\n")
end

function task_8a()
	v_data, v_y = svm_test_1()
	task_8(v_data, v_y)
	v_data, v_y = svm_test_2()
	task_8(v_data, v_y)
	v_data, v_y = svm_test_3()
	task_8(v_data, v_y)
	v_data, v_y = svm_test_4()
	task_8(v_data, v_y)
end

function task_8b(v_data,v_y)
	lambda = [0.1, 0.001, 0.00001]
	sigma = [2, 0.5, 0.25]
	x,y = svm_train()
	test_errors = zeros(length(lambda))
	train_errors = zeros(length(lambda))
	for l=1:length(lambda)
		println("lambda=$l \n sigma")
		v = svm(x, y, lambda[l], sigma[l])
		v = v[:,end]
		#print(sqrt.(sum(abs2, v, dims=1)))
		x_test_labels = zeros(length(v_data))
		for i=1:length(v_data)
			x_test_labels[i] = pred_model(v_data[i], x, y, v, lambda[l], sigma[l])
		end
		x_train_labels = zeros(length(x))
		for i=1:length(x)
			x_train_labels[i] = pred_model(x[i], x, y, v, lambda[l], sigma[l])
		end
		#v_y, true labels, x_hat_class, predicted labels
		#Training data error.
		test_error = mean(x_test_labels .!= v_y)
		train_error = mean(x_train_labels .!= y)
		test_errors[l] = test_error
		train_errors[l] = train_error
		println("test_error: $test_error")
		println("train_error: $train_error")
		println("\n")
	end
	return test_errors, train_errors, lambda, sigma
end

function task_8b2()
	v_data, v_y = svm_test_1()
	test_1 = task_8b(v_data, v_y)
	v_data, v_y = svm_test_2()
	test_2 = task_8b(v_data, v_y)
	v_data, v_y = svm_test_3()
	test_3 = task_8b(v_data, v_y)
	v_data, v_y = svm_test_4()
	test_4 = task_8b(v_data, v_y)
	return test_1, test_2, test_3, test_4
end

function task_8b_constants()
	v_data, v_y = svm_test_1()
	println(mean(v_y.==1))
	v_data, v_y = svm_test_2()
	println(mean(v_y.==1))
	v_data, v_y = svm_test_3()
	println(mean(v_y.==1))
	v_data, v_y = svm_test_4()
	println(mean(v_y.==1))
end

function task_9_runner(x, y, v_data, v_y, lambda, sigma)
	v = svm(x, y, lambda, sigma)
	v = v[:,end]
	#print(sqrt.(sum(abs2, v, dims=1)))
	x_test_labels = zeros(length(v_data))
	for i=1:length(v_data)
		x_test_labels[i] = pred_model(v_data[i], x, y, v, lambda, sigma)
	end
	x_train_labels = zeros(length(x))
	for i=1:length(x)
		x_train_labels[i] = pred_model(x[i], x, y, v, lambda, sigma)
	end
	#v_y, true labels, x_hat_class, predicted labels
	#Training data error.
	test_error = mean(x_test_labels .!= v_y)
	train_error = mean(x_train_labels .!= y)
	println("test_error: $test_error")
	println("train_error: $train_error")
	return test_error, train_error
end

function k_fold(lambda, sigma)
	mt = MersenneTwister(0xAC) # seed if we want to rexamine the same randperm.
	k = 10
	x,y = svm_train()
	N = length(x)
	indices = randperm(500)
	folds = fill(Int[], 10)
	for i=1:k
		folds[i] = indices[1+50(i-1):50*i]
	end
	# folds: 10 x 50.
	aggr_test_error = Float64[]
	aggr_train_error = Float64[]
	for i=1: length(folds)
		x_data = x[collect(Iterators.flatten(folds[1:end .!= i]))]
		y_data = y[collect(Iterators.flatten(folds[1:end .!= i]))]
		x_val = x[folds[i]]
		y_val = y[folds[i]]
		test_error, train_error = task_9_runner(x_data, y_data, x_val, y_val, lambda,sigma)
		push!(aggr_test_error, test_error)
		push!(aggr_train_error, train_error)
	end
	println("Average, 10 folds. Test error: ",mean(aggr_test_error))
	println("Average, 10 folds. Train error: ",mean(aggr_train_error))
	return mean(aggr_test_error), mean(aggr_train_error)
end

function hold_out()
	mt = MersenneTwister(0xAC) # seed if we want to rexamine the same randperm.
	x,y = svm_train()
	N = length(x)
	indices = randperm(500)
	#holdout cross validation
	x_train = x[indices[101:end]]
	y_train = y[indices[101:end]]
	x_val = x[indices[1:100]]
	y_val = y[indices[1:100]]
	test_error, train_error = task_9_runner(x_train, y_train, x_val, y_val)
end

function variance_fold(lambda, sigma)
	test_means = Float64[]
	train_means = Float64[]
	for i=1: 40
		test_error_mean, train_error_mean = k_fold(lambda,sigma)
		push!(test_means, test_error_mean)
		push!(train_means, train_error_mean)
	end
	var_test = var(test_means)
	var_train = var(train_means)
	println("Variance test error, k-fold (k=10)",var(test_means))
	println("Variance train error, k-fold (k=10)",var(train_means))
	fig2 = histogram(test_means, bins=25, title="Histogram, Hold-out. Mean=$(mean(test_means)) Variance: = $(round(var_test,digits=6))",legend=false, ylabel="Frequency", xlabel="Test error")
	savefig("histogram_k_fold_$(lambda)_$(sigma)")
	return test_means, train_means, var_test, var_train
end

function tuning()
	lambda = [0.1, 0.01, 0.001, 0.00001]
	sigma = [1, 0.75, 0.5, 0.25]
	for l=1:length(lambda)
		for s=1:length(sigma)
			println("lambda=$(lambda[l]), s=$(sigma[s])")
			variance_fold(lambda[l], sigma[s])
		end
	end
end

function variance_hold_out()
	test_means = Float64[]
	train_means = Float64[]
	for i=1: 100
		test_error_mean, train_error_mean = hold_out()
		push!(test_means, test_error_mean)
		push!(train_means, train_error_mean)
	end
	var_test = var(test_means)
	var_train = var(train_means)
	println("Variance test error, k-fold (k=10)",var(test_means))
	println("Variance train error, k-fold (k=10)",var(train_means))
	return test_means, train_means, var_test, var_train
end
