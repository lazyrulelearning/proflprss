
#include <cmath>
#include <vector>
#include <cstring>
#include <algorithm>

using namespace std;

typedef pair<int, int> prime_pot;

//MAX determines the primes limit
const unsigned MAX = 100010/60, MAX_S = sqrt(MAX/60);

unsigned w[16] = {1, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 49, 53, 59};
unsigned short composite[MAX];
vector<int> _primes;

//Generate primes with sieve
vector<int>& generate_primes()
{
    unsigned mod[16][16], di[16][16], num;

    for(int i = 0; i < 16; i++)
        for(int j = 0; j < 16; j++)
        {

            di[i][j] = (w[i] * w[j]) / 60;

            mod[i][j] = lower_bound(w, w + 16, (w[i] * w[j]) % 60) - w;
        }

    _primes.push_back(2);
    _primes.push_back(3);
    _primes.push_back(5);

    for(unsigned i = 0; i < MAX; i++)

        for(int j = (i == 0); j < 16; j++)
        {

            if(composite[i] & (1 << j)) continue;

            _primes.push_back(num = 60 * i + w[j]);

            if(i > MAX_S) continue;

            for(unsigned k = i, done = false; !done; k++)
                for(int l = (k == 0); l < 16 && !done; l++)
                {

                    unsigned mult = k * num + i * w[l] + di[j][l];

                    if(mult >= MAX) done = true;

                    else composite[mult] |= 1 << mod[j][l];
                }
        }
	return _primes;
}

