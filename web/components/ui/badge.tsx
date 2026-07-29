import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-semibold transition-colors [&_svg]:size-3.5",
  {
    variants: {
      variant: {
        default: "bg-primary/10 text-primary dark:bg-primary/20",
        secondary: "bg-secondary text-secondary-foreground",
        success: "bg-success/10 text-success dark:bg-success/20",
        destructive: "bg-destructive/10 text-destructive dark:bg-destructive/20",
        warning: "bg-accent/20 text-accent-foreground dark:text-accent",
        outline: "border text-foreground",
      },
    },
    defaultVariants: { variant: "default" },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge, badgeVariants };
