
def read_flag_x_times(shared_list, x=10, name='flag'):
    def tryAgain(retries=0):
        if retries < x:
            try:
                if shared_list[0]:
                    return True
                else:
                    return False
            except:
                tryAgain(retries+1)
        else:
            raise Exception('Could not read shared list: ', name, 'with value ', shared_list,' after ', x, ' tries.', flush=True)

    Flag = tryAgain()
    return Flag
