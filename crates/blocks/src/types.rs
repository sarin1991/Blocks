pub trait BlockConfig {}

pub trait BlockType {
    type Config:BlockConfig;
}

pub trait ViewRepr : BlockType {
    type View<'a> where Self:'a;
    fn view<'a>(&'a self) -> Self::View<'a>;
    fn validate<'a>(config: &Self::Config,view: &Self::View<'a>) -> bool;
}

pub trait ViewMutRepr : BlockType+ViewRepr {
    type ViewMut<'a> : BlockType<Config = Self::Config> + ViewRepr<View<'a> = Self::View<'a>>
    where Self:'a;
    fn view_mut<'a>(&'a mut self) -> Self::ViewMut<'a>;
    fn validate<'a>(config: &Self::Config,view: &Self::View<'a>) -> bool;
}