using Plots
using ProximalOperators
using LinearAlgebra
using Statistics
using Random

function svm(x, y, lambda, sigma, beta)
	iter = 100000
	println("Computing Kernel")
	K = kernel(x, sigma)
	Y = Diagonal(y)
	println("Computing Q")
	Q = (1/lambda) * (Y * K * Y)
	step_size = 1/opnorm(Q)
	mu = minimum(eigvals(Q))
	h = HingeLoss(ones(length(x)),1/length(x))
	h_conj = Conjugate(h)
	v = zeros(length(x), iter+1)
	println("Starting prox iterations")
	for i=2:iter+1
		if i==2
			grad = Q * v[:, i-1]
			v[:, i], _ = prox(h_conj, (v[:, i] - step_size*grad), step_size)
		else
			if beta == 1
				beta_value = (i-2)/(i+1)
			elseif beta == 2
				print("not implemented")
				exit()
			elseif beta == 3
				beta_value = (1-sqrt(mu*lambda))/(1+sqrt(mu*lambda))
			end
			v[:, i] = v[:, i-1] + beta_value*(v[:,i-1] - v[:, i-2])
			grad = Q * v[:, i]
			v[:, i], _ = prox(h_conj, (v[:,i] - step_size*grad), step_size)
		end
	end
	return v
end

function coordinate_prox()
	lambda = 0.001
	sigma = 0.5
	iter = 100000
	println("Computing Kernel")
	K = kernel(x, sigma)
	Y = Diagonal(y)
	println("Computing Q")
	Q = (1/lambda) * (Y * K * Y)
	step_size = 1/opnorm(Q)
	mu = minimum(eigvals(Q))
	h = HingeLoss(ones(length(x)),1/length(x))
	h_conj = Conjugate(h)
	v = zeros(length(x), iter+1)
	println("Starting prox iterations")
	for i=2:iter+1
		if i==2
			grad = Q * v[:, i-1]
			v[:, i], _ = prox(h_conj, (v[:, i] - step_size*grad), step_size)
		else
			if beta == 1
				beta_value = (i-2)/(i+1)
			elseif beta == 2
				print("not implemented")
				exit()
			elseif beta == 3
				beta_value = (1-sqrt(mu*lambda))/(1+sqrt(mu*lambda))
			end
			v[:, i] = v[:, i-1] + beta_value*(v[:,i-1] - v[:, i-2])
			grad = Q * v[:, i]
			v[:, i], _ = prox(h_conj, (v[:,i] - step_size*grad), step_size)
		end
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

function task_1(v_data,v_y, beta)
	lambda = 0.001
	sigma = 0.5
	x,y = svm_train()
	println("lambda=$lambda \n sigma=$sigma")
	v = svm(x, y, lambda, sigma, beta)
	v_plot = v
	v_opt = v[:,end]
	_, length = size(v)
	diff = v[:,2:length] .- v_opt
	norm_diff = sqrt.(sum(abs2, diff[:,1:end-1], dims=1))
	fig = plot(max.(1e-15,norm_diff'), legend=false,
		xlabel = "iteration nbr",
		title = "beta = $beta",
		yscale=:log10)
	savefig("svm_accelerated_beta$beta.png")
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

function k_fold()
	k = 10
	x,y = svm_train()
	N = length(x)
	indices = randperm(500)
	folds = fill(Int[], 10)
	for i=1:k
		folds[i] = indices[1+50(i-1):50*i]
	end
	# folds: 10 x 50.
	for i=1: length(folds)
		x_data = x[collect(Iterators.flatten(folds[1:end .!= i]))]
		y_data = y[collect(Iterators.flatten(folds[1:end .!= i]))]
		x_val = x[folds[i]]
		y_val = y[folds[i]]
		task_9_runner(x_data, y_data, x_val, y_val)
	end
end
