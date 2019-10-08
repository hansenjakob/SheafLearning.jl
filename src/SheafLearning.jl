module SheafLearning

greet() = print("Hello World!")

using LinearAlgebra
using Printf
using Optim

function obj(Me,Le::Array{Float64,3},alpha,beta,Nv,dv,t,barrier=true)
    #Check that the input is in the domain of the function; this ensures the line search works properly
    if !checkvalid(Le,1e-12)
        return Inf
    end

    trL = zeros(Nv)
    e = 1
    for i = 1:Nv
        for j = i+1:Nv
            indices = [dv*(i-1)+1:dv*i; dv*(j-1)+1:dv*j]
            trL[i] += tr(Le[1:dv,1:dv,e])
            trL[j] += tr(Le[dv+1:2*dv,dv+1:2*dv,e])
            e += 1
        end
    end

    obj = dot(Le,Me)
    obj += -alpha*sum(log.(trL))
    obj += beta*norm(Le[1:dv,dv+1:dv*2,:])^2
    if barrier
        obj *= t
        Ne = div(Nv*(Nv-1),2)
        for e = 1:Ne
            obj += -log(det(Le[:,:,e]))
        end
    end
    return obj
end

function obj_gradient!(grad::Array{Float64,3},Me,Le::Array{Float64,3},alpha,beta,Nv,dv,t)
    trL = zeros(Nv)
    e = 1
    for i = 1:Nv
        for j = i+1:Nv
            indices = [dv*(i-1)+1:dv*i; dv*(j-1)+1:dv*j]
            trL[i] += tr(Le[1:dv,1:dv,e])
            trL[j] += tr(Le[dv+1:2*dv,dv+1:2*dv,e])
            e += 1
        end
    end
    trLinv = (trL).^-1
    e = 1
    grad[:,:,:] = t*Me[:,:,:]
    for i = 1:Nv
        for j = i+1:Nv
            grad[1:dv,dv+1:2dv,e] += t*beta*Le[1:dv,dv+1:2dv,e]
            grad[dv+1:2dv,1:dv,e] += t*beta*Le[dv+1:2dv,1:dv,e] 
            for k = 1:dv
                grad[k,k,e] += -t*alpha*trLinv[i]
                grad[dv+k,dv+k,e] += -t*alpha*trLinv[j]
            end
            grad[:,:,e] += -inv(Le[:,:,e])
            e += 1
        end
    end
end


function checkvalid(Le::Array{Float64,3},eps)
    dim = size(Le)[1]
    Ne = size(Le)[3]
    for i = 1:Ne
        if det(Le[:,:,i]) <= eps
            return false
        end
        for k = 1:dim
            if Le[k,k,i] <= eps
                return false
            end
        end
    end
    return true
end

function LfromLe(Le,Nv,dv)
    L = zeros(Nv*dv,Nv*dv)
    e = 1
    for i = 1:Nv
        for j = i+1:Nv
            indices = [dv*(i-1)+1:dv*i; dv*(j-1)+1:dv*j]
            L[indices,indices] += Le[:,:,e]
            e += 1
        end
    end
    return L
end

function recover_sheaf_Laplacian(Me,alpha,beta,Nv,dv,tol=1e-7,maxouter=20,tscale=25)
    Le = zeros(2dv,2dv,Ne)
    for e = 1:Ne
        Le[:,:,e] = Matrix{Float64}(I, dv*2, dv*2)
    end
    t = 1
    oldobj = obj(Me,Le,alpha,beta,Nv,dv,t,false)
    m = length(Me)*2*dv
    for outeriter = 1:maxouter
        gradient_data! = (gradd,Lee) -> obj_gradient!(gradd,Me,Lee,alpha,beta,Nv,dv,t)
        obj_data = (Lee) -> obj(Me,Lee,alpha,beta,Nv,dv,t)
        res = optimize(obj_data,gradient_data!,Le,LBFGS())
        Le = Optim.minimizer(res)
        t = tscale * t
        newobj = obj(Me,Le,alpha,beta,Nv,dv,t,false)
        @printf("step %i objective: %f\n", outeriter, newobj)
        @printf("t = %f\n", t)
        if (m/t < tol)
            @printf("Desired tolerance %f reached\n",tol)
            break
        end
        oldobj = newobj
    end
    return Le
end
end # module
