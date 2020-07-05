class Bcolors:
  def __init__(self):
    self.HEADER     = '\033[95m'
    self.OKBLUE     = '\033[94m'
    self.OKGREEN    = '\033[92m'
    self.WARNING    = '\033[93m'
    self.FAIL       = '\033[91m'
    self.ENDC       = '\033[0m'
    self.BOLD       = '\033[1m'
    self.UNDERLINE  = '\033[4m'

  def print_header(self,msg):
    print(self.HEADER + msg + self.ENDC)
  def print_inform(self,msg):
    print(self.OKBLUE + msg + self.ENDC)
  def print_underline(self,msg):
    print(self.UNDERLINE + msg + self.ENDC)
  def print_bold(self,msg):
    print(self.BOLD + msg + self.ENDC)
  def print_ok(self,msg):
    print(self.OKGREEN + msg + self.ENDC)
  def print_warning(self,msg):
    print(self.WARNING + msg + self.ENDC)
  def print_error(self,msg):
    print(self.FAIL + msg + self.ENDC)
