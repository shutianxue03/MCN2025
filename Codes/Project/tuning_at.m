    function [f, fp] = tuning_at(tuningDiversity, s, pref_all, a, b, kappa, a_all, b_all, kappa_all)
        % per-neuron von Mises centered at pref_all
        d = s - pref_all;  % N x 1
        switch tuningDiversity
            case 'uniform'
                f  = b + a .* exp( kappa    .* (cos(d) - 1) );
                fp = -a .* kappa .* exp( kappa .* (cos(d) - 1) ) .* sin(d);
            case 'lowDiversity'
                f  = b + a .* exp( kappa_all .* (cos(d) - 1) );
                fp = -a .* kappa_all .* exp( kappa_all .* (cos(d) - 1) ) .* sin(d);
            case 'naturalDiversity'
                f  = b_all + a_all .* exp( kappa_all .* (cos(d) - 1) );
                fp = -a_all .* kappa_all .* exp( kappa_all .* (cos(d) - 1) ) .* sin(d);
            otherwise
                error('Unknown tuningDiversity');
        end
        f = max(f, 1e-6);  % keep strictly positive
    end