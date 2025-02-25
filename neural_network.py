# دالة لحساب الـ exp(x) باستخدام متسلسلة تايلور
def exp(x, terms=10):
    result = 1.0
    term = 1.0
    for i in range(1, terms):
        term *= x / i
        result += term
    return result

# دالة لحساب tanh(x) بدون مكتبات
def tanh(x):
    exp_x = exp(x)
    exp_neg_x = exp(-x)
    return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)

# مولد أرقام عشوائية بسيط بدون مكتبات (Linear Congruential Generator)
seed = 42
def random_uniform(low, high):
    global seed
    seed = (1664525 * seed + 1013904223) % (2**32)
    return low + (seed / (2**32)) * (high - low)

# دالة تهيئة الأوزان العشوائية
def initialize_weights():
    return [random_uniform(-0.5, 0.5) for _ in range(8)]

# التمرير الأمامي في الشبكة العصبية
def forward_pass(i1, i2, w, b1, b2, b3, b4):
    # الطبقة المخفية
    h1_input = i1 * w[0] + i2 * w[1] + b1
    h2_input = i1 * w[2] + i2 * w[3] + b2
    h1_output = tanh(h1_input)
    h2_output = tanh(h2_input)
    
    # الطبقة الإخراجية
    o1_input = h1_output * w[4] + h2_output * w[5] + b3
    o2_input = h1_output * w[6] + h2_output * w[7] + b4
    o1_output = tanh(o1_input)
    o2_output = tanh(o2_input)
    
    return o1_output, o2_output

# المدخلات
i1, i2 = 0.05, 0.10
# تهيئة الأوزان والانحيازات
weights = initialize_weights()
b1, b2 = 0.5, 0.5  # انحيازات الطبقة المخفية
b3, b4 = 0.7, 0.7  # انحيازات الطبقة الإخراجية
# حساب المخرجات
o1, o2 = forward_pass(i1, i2, weights, b1, b2, b3, b4)
# طباعة النتائج
print(f"Output o1: {o1:.4f}, Output o2: {o2:.4f}")
