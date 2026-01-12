from threading import Lock


class BasicMetrics:
    def __init__(self):
        self.lock = Lock()
        self.total_request = 0
        self.total_erros = 0
        self.total_sucess = 0

        self.max_latency_sofar = 0
        self.last_latency_ms = None
        self.erros_rate = 0


    def inc_reuqest(self):
        with self.lock:
            self.total_request +=1
    
    def inc_sucess(self):
        with self.lock:
            self.total_sucess +=1

    def inc_erros(self):
        with self.lock:
            self.total_erros +=1

    def record_latency(self, last_latency_ms):
        with self.lock:
            self.last_latency_ms = last_latency_ms
            if last_latency_ms > self.max_latency_sofar:
                self.max_latency_sofar = last_latency_ms

                


    def display_snapshot(self):
        with self.lock:
            total_request = self.total_request
            total_erros = self.total_erros

            if total_request > 0:
                self.erros_rate = (total_erros)/(total_request) * 100 
            else:
                self.erros_rate = 0.0
            return {
                "total_request" : self.total_request,
                "total_sucess" : self.total_sucess,
                "total_erros" : self.total_erros,
                "last_latency_ms" : self.last_latency_ms,
                "max_latency_sofar" : self.max_latency_sofar,
                "error_rate" : f"{round(self.erros_rate,4)} %"
            }



Obj_Basic_Metrics = BasicMetrics()