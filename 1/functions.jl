"""
    quad(x,Q,q)

Compute the quadratic

	1/2 x'Qx + q'x

"""
function quad(x,Q,q)
	return 1/2 * x'*Q*x + q'*x
end



"""
    guadconj(y,Q,q)

Compute the convex conjugate of the quadratic

	1/2 x'Qx + q'x

"""
function quadconj(y,Q,q)
	1/2 * (y - q)'inv(Q)(y - q)
end



"""
    box(x,a,b)

Compute the indicator function of for the box contraint

	a <= x <= b

where the inequalites are applied element-wise.
"""
function box(x,a,b)
	return all(a .<= x .<= b) ? 0.0 : Inf
end



"""
    boxconj(y,a,b)

Compute the convex conjugate of the indicator function of for the box contraint

	a <= x <= b

where the inequalites are applied element-wise.
"""
function boxconj(y,a,b)
	res = 0
	for x=1: length(y)
		if(y[x] <= 0)
			res += y[x]*a[x]
		else
			res += y[x]*b[x]
		end
	end
	return res
end



"""
    grad_quad(x,Q,q)

Compute the gradient of the quadratic

	1/2 x'Qx + q'x

"""
function grad_quad(x,Q,q)
	return Q*x + q
end



"""
    grad_quadconj(y,Q,q)

Compute the gradient of the convex conjugate of the quadratic

	1/2 x'Qx + q'x

"""
function grad_quadconj(y,Q,q)
	inv(Q) * (y - q)
end



"""
    prox_box(x,a,b)

Compute the proximal operator of the indicator function for the box contraint

	a <= x <= b

where the inequalites are applied element-wise.
"""
function prox_box(x,a,b,gamma)
	res = similar(x)
	for i=1: length(x)
		if(x[i] < a[i])
			res[i] = a[i]
		elseif(x[i] > b[i])
			res[i] = b[i]
		else
			res[i] = x[i]
		end
	end
	return res
end

"""
    prox_boxconj(y,a,b)

Compute the proximal operator of the convex conjugate of the indicator function
for the box contraint

	a <= x <= b

where the inequalites are applied element-wise.
"""
function prox_boxconj(y,a,b,gamma)
	res = similar(y)
	for i=1: length(y)
		if(y[i] < gamma*a[i])
			res[i] = y[i] - gamma*a[i]
		elseif(y[i] > gamma*b[i])
			res[i] = y[i] - gamma*b[i]
		else
			res[i] = 0
		end
	end
	return res
end


"""
    dual2primal(y,Q,q,a,b)

Computes the solution to the primal problem for Hand-In 1 given a solution y to
the dual problem.
"""
function dual2primal(y, Q, q, a, b)
	_, length = size(y)
	all_solutions = similar(y)
	for i=1:length
		all_solutions[:,i] = grad_quadconj(y[:,i],Q,q)
	end
	return all_solutions
end
