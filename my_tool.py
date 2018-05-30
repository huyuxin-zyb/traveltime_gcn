import datetime


class myDate:
    def __init__(self, day):
        if len(day)>11:
            self.day=datetime.datetime.strptime(day, '%Y-%m-%d %H:%M:%S')
        else:
            self.day = datetime.datetime.strptime(day, '%Y-%m-%d')

    def get_date(self):
        return self.day

    def get_bin(self,min=30):
        return self.day.hour*(60/min)+ self.day.minute/min+1

    def get_day(self,x_1=False):
        if x_1:
            return datetime.date.isoweekday(self.day)
        if datetime.date.isoweekday(self.day)<5:
            return 1
        if datetime.date.isoweekday(self.day)==5:
            return 2
        return 3

    def get_ago(self,delta):
        delta=datetime.timedelta(days=delta)
        return (self.day-delta).strftime('%Y-%m-%d %H:%M:%S')


def mergesort(seq):
    if len(seq) <= 1:
        return seq
    mid = int(len(seq) / 2)
    left = mergesort(seq[:mid])
    right = mergesort(seq[mid:])
    return merge(left, right)


def merge(left, right):
    result = []
    i, j = 0, 0
    while i < len(left) and j < len(right):
        if left[i][0] <= right[j][0]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result += left[i:]
    result += right[j:]
    return result
